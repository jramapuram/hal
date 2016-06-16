use af;
use af::{Array, Dim4, DType};
use std::cell::{RefCell, Cell};

use utils;
use data::{Data, DataSource, DataParams, Normalize, Shuffle};

pub struct SinSource {
  pub params: DataParams,
  pub iter: Cell<u64>,
  pub offset: Cell<f32>,
}

impl SinSource {
  pub fn new(input_size: u64, batch_size: u64, dtype: DType
             , max_samples: u64, is_normalized: bool
             , is_shuffled: bool) -> SinSource
  {
    let dims = Dim4::new(&[batch_size, input_size, 1, 1]);
    let train_samples = 0.7 * max_samples as f32;
    let test_samples = 0.2 * max_samples as f32;
    let validation_samples = 0.1 * max_samples as f32;
    SinSource {
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
      offset: Cell::new(0.0f32),
    }
  }

  fn generate_sin_wave(&self, input_dims: u64, num_rows: u64) -> Array {
    let tdims = Dim4::new(&[input_dims, num_rows, 1, 1]);
    let dims = Dim4::new(&[1, num_rows * input_dims, 1, 1]);
    let x = af::transpose(&af::moddims(&af::range::<f32>(dims, 1)
                                       , tdims), false);
    let x_shifted = af::add(&self.offset.get()
                            , &af::div(&x, &(input_dims), false)
                            , false);
    self.offset.set(self.offset.get() + 1.0/(input_dims*num_rows - 1) as f32);
    utils::cast(&af::sin(&x_shifted), self.params.dtype)
  }
}

impl DataSource for SinSource
{
  fn get_train_iter(&self, num_batch: u64) -> Data {
    let inp = self.generate_sin_wave(self.params.input_dims[1], num_batch);
    let mut batch = Data {
      input: RefCell::new(Box::new(inp.clone())),
      target: RefCell::new(Box::new(inp.copy())),
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
