pub use self::sin::SinSource;
mod sin;

pub use self::xor::XORSource;
mod xor;

unsafe impl Send for SinSource {}
unsafe impl Sync for SinSource {}

// Doesn't make sense to allow sequential sources to be asynchronously polled, right?
// unsafe impl Send for XORSource {}
// unsafe impl Sync for XORSource {}

use af::{Dim4, Array, DType};
use std::cell::{RefCell, Cell};
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;
use std::collections::VecDeque;
use std::time::Duration;
use spmc::{Sender, TryRecvError, channel};

use device::{DeviceManager, Device};
use utils;

#[derive(Clone)]
pub struct Data {
  pub input: RefCell<Box<Array>>,
  pub target: RefCell<Box<Array>>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct DataParams {
  pub input_dims: Dim4,         // [batch_size, feature_x, feature_y, time]
  pub target_dims: Dim4,        // [batch_size, feature_x, feature_y, time]
  pub shuffle: bool,            // whether the data is shuffled
  pub normalize: bool,          // whether the data is normalized
  pub current_epoch: Cell<u64>, // for internal tracking of what the current epoch is
  pub dtype: DType,             // the type of data generated
  pub num_samples: u64,
  pub num_train: u64,
  pub num_test: u64,
  pub num_validation: Option<u64>,
}

/// A DataLoader spawns helper threads in order to load data asynchronously
/// Test & Validation buffers are filled lazily (on first query)
///
/// # Parameters
/// - `num_threads` are the number of threads that keep acquiring data
/// - `manager` is the device manager
/// - `device` is the source data device
/// - `num_batch` is the locked batch size for this data loader
/// - `train_buffer_size` is the size of the training buffer
/// - `test_buffer_size` is the size of the testing buffer
/// - `validation_buffer_size` is the size of the validation buffer
/// - `source` is the input datasource to mask
/// - `queue_train` is the deque containing the training buffer
/// - `queue_test` is the deque containing the testing buffer
/// - `queue_validation` is the deque containing the validation buffer
/// - `test_query_started` is the flag that is set to start filling the test buffer
/// - `validation_query_started` is the flag that is set to start filling the validation buffer
/// - `is_terminated` is the channel used to stop the worker threads
/// - `handles` are the thread handles used to join on a drop
pub struct DataLoader {
  pub num_threads: u64,
  pub manager: DeviceManager,
  pub device: Device,
  pub num_batch: u64,
  pub train_buffer_size: u64,
  pub test_buffer_size: u64,
  pub validation_buffer_size: u64,
  pub source: Arc<Mutex<DataSource + Sync + Send>>,
  pub queue_train: Arc<Mutex<VecDeque<Data>>>,
  pub queue_test: Arc<Mutex<VecDeque<Data>>>,
  pub queue_validation: Arc<Mutex<VecDeque<Data>>>,
  pub test_query_started: Arc<Mutex<bool>>,
  pub validation_query_started: Arc<Mutex<bool>>,
  pub is_terminated: Sender<i32>,
  pub handles: Vec<Option<JoinHandle<()>>>,
}

/// Helper method to call the lambda function to fill the provided queue
fn fill_queue<F>(fn_closure: F, src: Arc<Mutex<DataSource + Sync + Send>>
                 , manager: DeviceManager, device: Device
                 , q: Arc<Mutex<VecDeque<Data>>>, max_queue_length: u64)
  where F : Fn(Arc<Mutex<DataSource + Sync + Send>>) -> Option<Data>
{
  let mut qtex = &mut q.lock().unwrap();
  //println!("q len is {}", qtex.len());
  if qtex.len() < max_queue_length as usize {
    let current_device = manager.current_device();
    if current_device.id != device.id
      || current_device.backend != device.backend
    {
      manager.swap_device(device);
    }
    let current_data = fn_closure(src.clone());
    match current_data {
      Some(data) => qtex.push_front(data),
      None       => (),
    };
    manager.swap_device(current_device);
  }
}

impl Drop for DataLoader {
  fn drop(&mut self) {
    self.is_terminated.send(-1).unwrap();
    for handle in self.handles.iter_mut() {
      handle.take().unwrap().join().unwrap();
    }
  }
}

fn wait_and_sleep(q: Arc<Mutex<VecDeque<Data>>>)
{
  let mut is_empty = true;
  while is_empty {
    { // scope out mutex
      let qtex = q.lock().unwrap();
      is_empty = qtex.is_empty();
    }
    // wait without locking
    if is_empty {
      thread::sleep(Duration::from_millis(50)); // XXX
    }
  }
}

impl DataLoader {
  pub fn new(num_threads: u64
             , manager: DeviceManager
             , device: Device
             , train_buffer_size: u64
             , test_buffer_size: u64
             , validation_buffer_size: u64
             , num_batch: u64
             , source: Arc<Mutex<DataSource + Sync + Send>>) -> DataLoader
  {
    // setup the single producer multiple consumer termination channel
    let (tx, rx) = channel();

    // avoid passing 'self' to the thread
    let train = Arc::new(Mutex::new(VecDeque::new()));
    let test = Arc::new(Mutex::new(VecDeque::new()));
    let validation = Arc::new(Mutex::new(VecDeque::new()));
    let test_started = Arc::new(Mutex::new(false));
    let validation_started = Arc::new(Mutex::new(false));

    // spin up the threads
    let mut threads = Vec::new();
    for thread_num in 0..num_threads {
      let q_train_arc = train.clone();
      let q_test_arc = test.clone();
      let q_valid_arc = validation.clone();
      let rx = rx.clone();
      let test_started_arc = test_started.clone();
      let validation_started_arc = validation_started.clone();
      let src = source.clone();
      let manager = manager.clone();

      threads.push(Some(thread::spawn(move || {
        println!("starting data worker thread {}...", thread_num);
        loop {
          let test_mtx = test_started_arc.lock().unwrap();
          let valid_mtx = validation_started_arc.lock().unwrap();

          // gather in test data if it is required
          if *test_mtx {
            let lambda = |x: Arc<Mutex<DataSource + Sync + Send>>| {
              let x = x.lock().unwrap();
              Some(x.get_test_iter(num_batch))
            };
            fill_queue(lambda, src.clone(), manager.clone(), device.clone(), q_test_arc.clone()
                       , test_buffer_size);
          }

          // gather in validation data if it is required
          if *valid_mtx {
            let lambda = |x: Arc<Mutex<DataSource + Sync + Send>>| {
              let x = x.lock().unwrap();
              x.get_validation_iter(num_batch)
            };
            fill_queue(lambda, src.clone(), manager.clone(), device.clone(), q_valid_arc.clone()
                       , validation_buffer_size);
          }

          // always fill in training data
          let lambda = |x: Arc<Mutex<DataSource + Sync + Send>>| {
            let x = x.lock().unwrap();
            Some(x.get_train_iter(num_batch))
          };
          fill_queue(lambda, src.clone(), manager.clone(), device.clone(), q_train_arc.clone()
                     , train_buffer_size);

          match rx.try_recv() {
            Ok(_) | Err(TryRecvError::Disconnected) => {
              println!("terminating data worker {}...", thread_num);
              break;
            },
            Err(TryRecvError::Empty) => {},
          }
        }
      })));
    }

    DataLoader {
      num_threads: num_threads,
      manager: manager.clone(),
      device: device.clone(),
      num_batch: num_batch,
      train_buffer_size: train_buffer_size,
      test_buffer_size: test_buffer_size,
      validation_buffer_size: validation_buffer_size,
      source: source.clone(),
      test_query_started: test_started,
      validation_query_started: validation_started,
      queue_train: train,
      queue_test: test,
      queue_validation: validation,
      is_terminated: tx,
      handles: threads,
    }
  }
}

impl DataSource for DataLoader {
  fn info(&self) -> DataParams {
    let src = self.source.clone();
    let stex = src.lock().unwrap();
    stex.info()
  }

  fn get_train_iter(&self, num_batch: u64) -> Data{
    assert!(self.num_batch == num_batch
            , "batch sizes are currently non-mandible for the dataloader");
    let arc = self.queue_train.clone();
    wait_and_sleep(arc.clone());

    let mut q = &mut arc.lock().unwrap();
    q.pop_back().unwrap()
  }

  fn get_test_iter(&self, num_batch: u64) -> Data{
    assert!(self.num_batch == num_batch
            , "batch sizes are currently non-mandible for the dataloader");
    {
      let mut test_flag = self.test_query_started.lock().unwrap();
      *test_flag = true;
    }
    let arc = self.queue_test.clone();
    wait_and_sleep(arc.clone());

    let mut q = &mut arc.lock().unwrap();
    q.pop_back().unwrap()
  }

  fn get_validation_iter(&self, num_batch: u64) -> Option<Data>{
    assert!(self.num_batch == num_batch
            , "batch sizes are currently non-mandible for the dataloader");
    {
      let mut validation_flag = self.validation_query_started.lock().unwrap();
      *validation_flag = true;
    }
    let arc = self.queue_validation.clone();
    wait_and_sleep(arc.clone());

    let mut q = &mut arc.lock().unwrap();
    q.pop_back()
  }
}

/// A DataSource needs to provide these basic features
///
/// 1) It gives information regarding the source
/// 2) It provides a train iterator that returns a minibatch
/// 3) It provides a test iterator that returns a minibatch
/// 4) It (optionally)provides a validation iterator that returns a minibatch
pub trait DataSource {
  fn info(&self) -> DataParams;
  fn get_train_iter(&self, num_batch: u64) -> Data;
  fn get_test_iter(&self, num_batch: u64) -> Data;
  fn get_validation_iter(&self, num_batch: u64) -> Option<Data>;
}

/// Trait that describes a normalization operation
pub trait Normalize {
  fn normalize(&mut self, num_std: f32);
}

// TODO: Implement whitening via SVD
// pub trait Whiten {
//   fn whiten(&self);
// }

/// Trait that describes a shuffling operation
pub trait Shuffle {
  fn shuffle(&mut self);
}


// TODO: Implement this
impl Shuffle for Data {
  fn shuffle(&mut self) {
    //utils::shuffle_array(&mut[data.input.m, target], idims[0])
    //data.clone()
    println!("WARNING: shuffle not yet implemented");
  }
}

/// Implementation of the Normalize operation for Data
///
/// Currently only mean subtraction & std-deviation
/// division are supported. In the future we can
/// add whitening, etc.
impl Normalize for Data {
  fn normalize(&mut self, num_std: f32){
    let normalized_inputs = utils::normalize_array(&self.input.borrow(), num_std);
    let normalized_target = utils::normalize_array(&self.target.borrow(), num_std);
    self.input = RefCell::new(Box::new(normalized_inputs));
    self.target = RefCell::new(Box::new(normalized_target));
  }
}
