extern crate rand;
use rand::distributions::{Range, IndependentSample};
use af;
use af::{Array, Dim4, MatProp, DType};
use std::cell::{RefCell, Cell};

use initializations::uniform;
use utils;

use data::{Data, DataSource, DataParams, Normalize, Shuffle};

pub struct AddingProblemSource {
  pub params: DataParams,
  pub iter: Cell<u64>,
  pub offset: Cell<f32>,
  pub bptt_unroll : u64,
}

impl AddingProblemSource {
  pub fn new(batch_size: u64, bptt_unroll: u64, dtype: DType, max_samples: u64) -> AddingProblemSource
  {
    assert!(bptt_unroll % 4 == 0, "The number of time steps has to be divisible by 4 for the adding problem");
    let input_dims = Dim4::new(&[batch_size, 1, bptt_unroll, 1]);
    let target_dims = Dim4::new(&[batch_size, 1, bptt_unroll, 1]);
    let train_samples = 0.7 * max_samples as f32;
    let test_samples = 0.2 * max_samples as f32;
    let validation_samples = 0.1 * max_samples as f32;
    AddingProblemSource {
      params : DataParams {
        input_dims: input_dims,
        target_dims: target_dims,
        dtype: dtype,
        normalize: false,
        shuffle: false,
        current_epoch: Cell::new(0),
        num_samples: max_samples,
        num_train: train_samples as u64,
        num_test: test_samples as u64,
        num_validation: Some(validation_samples as u64),
      },
      iter: Cell::new(0),
      offset: Cell::new(0.0f32),
      bptt_unroll: bptt_unroll,
    }
  }

  fn generate_input(&self, batch_size: u64, bptt_unroll: u64) -> Array {
    let dim1 = Dim4::new(&[batch_size, 1, bptt_unroll/2, 1]);
    let ar1 = uniform::<f32>(dim1,-1.0, 1.0);

    let between1 = Range::new(0, bptt_unroll/4);
    let between2 = Range::new(bptt_unroll/4, bptt_unroll/2);
    let mut rng1 = rand::thread_rng();
    let mut rng2 = rand::thread_rng();

    let mut vec_total = Vec::with_capacity((batch_size*bptt_unroll) as usize);
    let vec_zeros = vec!(0f32; (bptt_unroll/2) as usize);

    for _ in 0..batch_size {
      let index1 = between1.ind_sample(&mut rng1) as usize;
      let index2 = between2.ind_sample(&mut rng2) as usize;
      let mut vec_temp = vec_zeros.clone();
      vec_temp[index1] = 1f32;
      vec_temp[index2] = 1f32;
      vec_total.extend(vec_temp);
    }
    let dim2 = Dim4::new(&[bptt_unroll/2, batch_size, 1, 1]);
    let ar2 = af::moddims(&af::transpose(&utils::vec_to_array::<f32>(vec_total, dim2), false), dim1);
    af::join(2, &ar1, &ar2)
  }

  fn generate_target(&self, input: &Array, batch_size: u64, bptt_unroll: u64) -> Array {
    let first = af::slices(&input, 0, bptt_unroll/2-1);
    let second = af::slices(&input, bptt_unroll/2, bptt_unroll-1);
    let ar = af::mul(&first, &second, false);

    let zeros = af::constant(0f32, Dim4::new(&[batch_size, 1, bptt_unroll, 1]));
    af::add(&af::sum(&ar, 2), &zeros, true)
  }
}

impl DataSource for AddingProblemSource {
  fn get_train_iter(&self, num_batch: u64) -> Data {
    let inp = self.generate_input(num_batch
                                  , self.params.input_dims[2]);
    let tar = self.generate_target(&inp
                                   , num_batch
                                   , self.params.input_dims[2]);
    let batch = Data {
      input: RefCell::new(Box::new(inp.clone())),
      target: RefCell::new(Box::new(tar.clone())),
    };

    let current_iter = self.params.current_epoch.get();
    if self.iter.get() == self.params.num_samples as u64/ num_batch as u64 {
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
    Some( self.get_train_iter(num_batch))
  }
}
