extern crate rand;
use rand::distributions::{Range, IndependentSample};
use af;
use af::{Array, Dim4, DType};
use std::cell::{RefCell, Cell};

use initializations;
use utils;

use data::{Data, DataSource, DataParams, Normalize, Shuffle};

pub struct CopyingProblemSource {
  pub params: DataParams,
  pub iter: Cell<u64>,
  pub offset: Cell<f32>,
  pub bptt_unroll : u64, 
  pub seq_size : u64,
}

impl CopyingProblemSource {
  pub fn new(input_size: u64, batch_size: u64, seq_size: u64
             , bptt_unroll: u64, dtype: DType, max_samples: u64) -> CopyingProblemSource
  {
    assert!(bptt_unroll > 2*seq_size, "The number of time steps has to be bigger than the sequence size x2 for the copying problem");
    let input_dims = Dim4::new(&[batch_size, input_size, bptt_unroll, 1]);
    //let output_dims = Dim4::new(&[batch_size, output_size, bptt_unroll, 1]);
    let train_samples = 0.7 * max_samples as f32;
    let test_samples = 0.2 * max_samples as f32;
    let validation_samples = 0.1 * max_samples as f32;
    CopyingProblemSource {
      params : DataParams {
        input_dims: input_dims,
        target_dims: input_dims,
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
      seq_size: seq_size,
    }
  }

  fn generate_input(&self, batch_size: u64, input_size: u64, bptt_unroll: u64, seq_size: u64) -> Array {

    let between = Range::new(0,input_size-2);
    let mut rng = rand::thread_rng();

    let mut vec_total = Vec::with_capacity((batch_size*input_size*bptt_unroll) as usize);
    let vec_zeros = vec!(0f32; input_size as usize);

    for _ in 0..seq_size {
      for _ in 0..batch_size {
        let index = between.ind_sample(&mut rng) as usize;
        let mut vec_temp = vec_zeros.clone();
        vec_temp[index] = 1f32;
        vec_total.extend(vec_temp);
      }
    }

    for _ in 0..bptt_unroll-2*seq_size-1 {
      for _ in 0..batch_size {
        let mut vec_temp = vec_zeros.clone();
        vec_temp[(input_size-2) as usize] = 1f32;
        vec_total.extend(vec_temp);
      }
    }

    for _ in 0..batch_size {
      let mut vec_temp = vec_zeros.clone();
      vec_temp[(input_size-1) as usize] = 1f32;
      vec_total.extend(vec_temp);
    }

    for _ in 0..seq_size {
      for _ in 0..batch_size {
        let mut vec_temp = vec_zeros.clone();
        vec_temp[(input_size-2) as usize] = 1f32;
        vec_total.extend(vec_temp);
      }
    }

    let ar_dims = Dim4::new(&[input_size, batch_size, bptt_unroll, 1]);
    af::transpose(&utils::vec_to_array::<f32>(vec_total, ar_dims), false)
  }

  fn generate_target(&self, input: &Array, batch_size: u64, input_size: u64, bptt_unroll: u64, seq_size: u64) -> Array {

    let mut vec_total = Vec::with_capacity((batch_size*input_size*(bptt_unroll-seq_size)) as usize);
    let vec_zeros = vec!(0f32; input_size as usize);

    for _ in 0..bptt_unroll-seq_size{
      for _ in 0..batch_size {
        let mut vec_temp = vec_zeros.clone();
        vec_temp[(input_size-2) as usize] = 1f32;
        vec_total.extend(vec_temp);
      }
    }

    let dim_first = af::Dim4::new(&[input_size,batch_size, bptt_unroll-seq_size,1]);
    let first = af::transpose(&utils::vec_to_array::<f32>(vec_total, dim_first), false);

    let second = af::slices(&input, 0, seq_size-1);
    af::join(2, &first, &second)
  }
}

impl DataSource for CopyingProblemSource {
  fn get_train_iter(&self, num_batch: u64) -> Data {
    let inp = self.generate_input(num_batch
                                  , self.params.input_dims[1]
                                  , self.params.input_dims[2]
                                  , self.seq_size);
    let tar = self.generate_target(&inp
                                   , num_batch
                                   , self.params.target_dims[1]
                                   , self.params.target_dims[2]
                                   , self.seq_size);
    let mut batch = Data {
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














