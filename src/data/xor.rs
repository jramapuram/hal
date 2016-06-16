use af::{Array, Dim4};
use std::cell::{RefCell, Cell};
use rand;
use rand::distributions::{IndependentSample, Range};

use utils;
use data::{Data, DataSource, DataParams, Normalize, Shuffle};

pub struct XORSource {
  pub params: DataParams,
  pub iter: Cell<u64>,
  pub last_x: Cell<f32>,
}

impl XORSource {
  pub fn new(input_size: u64, batch_size: u64, seq_len: u64
             , max_samples: u64, is_normalized: bool
             , is_shuffled: bool) -> XORSource
  {
    let dims = Dim4::new(&[batch_size, input_size, seq_len, 1]);
    let train_samples = 0.7 * max_samples as f32;
    let test_samples = 0.2 * max_samples as f32;
    let validation_samples = 0.1 * max_samples as f32;
    XORSource {
      params: DataParams {
        input_dims: dims,   // input is the same size as the output
        target_dims: dims,  // ^
        normalize: is_normalized,
        shuffle: is_shuffled,
        current_epoch: Cell::new(0),
        num_samples: max_samples,
        num_train: train_samples as u64,
        num_test: test_samples as u64,
        num_validation: Some(validation_samples as u64),
      },
      iter: Cell::new(0),
      last_x: Cell::new(0f32),
    }
  }

  pub fn generate_minibatch(&self, batch_size: u64) -> (Array, Array){
    // generator for x
    let between = Range::new(0i32, 2i32); //range is non-inclusive
    let mut rng = rand::thread_rng();

    let input_size = self.params.input_dims[1];
    let seq_size = self.params.input_dims[2];
    let total_size = batch_size * seq_size * input_size;
    let mut x: Vec<f32> = vec![0f32; total_size as usize];
    let mut y: Vec<f32> = vec![0f32; total_size as usize];
    for i in 0..total_size as usize
    {
      x[i] = between.ind_sample(&mut rng) as f32;
      y[i] = (self.last_x.get() as i32 ^ x[i] as i32) as f32;
      self.last_x.set(x[i]);
    }

    (utils::vec_to_array(x, self.params.input_dims),
     utils::vec_to_array(y, self.params.target_dims))
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
