use af;
use rand;
use rand::distributions::{IndependentSample, Range};
use af::{Array, Dim4, DType};
use std::sync::{Arc, Mutex};
use std::cell::{RefCell, Cell};

use utils;
use data::{Data, DataSource, DataParams, Normalize, Shuffle};

pub struct XORSource {
  pub params: DataParams,
  pub iter: Cell<u64>,
  pub last_x: Arc<Mutex<Array>>,
}

impl XORSource {
  pub fn new(input_size: u64, batch_size: u64, seq_len: u64
             , dtype: DType, max_samples: u64
             , is_normalized: bool, is_shuffled: bool) -> XORSource
  {
    let dims = Dim4::new(&[batch_size, input_size, seq_len, 1]);
    let train_samples = 0.7 * max_samples as f32;
    let test_samples = 0.2 * max_samples as f32;
    let validation_samples = 0.1 * max_samples as f32;
    XORSource {
      params: DataParams {
        input_dims: dims,   // input is the same size as the output
        target_dims: dims,  // ^
        dtype: dtype,
        normalize: is_normalized,
        shuffle: is_shuffled,
        current_epoch: Cell::new(0),
        num_samples: max_samples,
        num_train: train_samples as u64,
        num_test: test_samples as u64,
        num_validation: Some(validation_samples as u64),
      },
      iter: Cell::new(0),
      last_x: Arc::new(Mutex::new(utils::constant(dims, DType::B8, 0.0f32))),
    }
  }

  pub fn generate_minibatch(&self, batch_size: u64) -> (Array, Array){
    // grab an arc and unlock the mutex
    let last_x = self.last_x.clone();
    let mut lastex = last_x.lock().unwrap();

    assert!(batch_size == lastex.dims()[0]
            ,"xorsource does not allow varying batch sizes");
    let dims = Dim4::new(&[batch_size, self.params.input_dims[1]
                           , self.params.input_dims[2], self.params.input_dims[3]]);

    // generate the random type
    //let x_t = utils::constant(self.params.input_dims, self.params.dtype, 0.0f32);
    let x_t = af::randu::<bool>(dims);
    let y_t = af::bitxor(&x_t, &lastex);

    *lastex = x_t.clone();
    (utils::cast(&x_t, self.params.dtype)
     , utils::cast(&y_t, self.params.dtype))
  }
}

impl DataSource for XORSource
{
  fn get_train_iter(&self, num_batch: u64) -> Data {
    let (inps, tars) = self.generate_minibatch(num_batch);
    let mut batch = Data {
      input: RefCell::new(Box::new(inps.clone())),
      target: RefCell::new(Box::new(tars.copy())),
    };

    if self.params.normalize { batch.normalize(1.0); }
    if self.params.shuffle   {  batch.shuffle(); }
    let current_iter = self.params.current_epoch.get();
    if self.iter.get()  == self.params.num_samples as u64/ num_batch as u64 {
      self.params.current_epoch.set(current_iter + 1);
      self.iter.set(0);
    }
    self.iter.set(self.iter.get() + 1);
    batch
  }

  fn info(&self) -> DataParams {
    self.params.clone()
  }

  fn get_test_iter(&self, num_batch: u64) -> Data {
    self.get_train_iter(num_batch)
  }

  fn get_validation_iter(&self, num_batch: u64) -> Option<Data> {
    Some(self.get_train_iter(num_batch))
  }
}
