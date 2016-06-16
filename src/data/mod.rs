pub use self::sin::SinSource;
mod sin;

pub use self::xor::XORSource;
mod xor;

use af::{Dim4, Array};
use std::cell::{RefCell, Cell};

use utils;

#[derive(Clone)]
pub struct Data {
  pub input: RefCell<Box<Array>>,
  pub target: RefCell<Box<Array>>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct DataParams {
  pub input_dims: Dim4,   // [batch_size, feature_x, feature_y, time]
  pub target_dims: Dim4,  // [batch_size, feature_x, feature_y, time]
  pub shuffle: bool,      // whether the data is shuffled
  pub normalize: bool,    // whether the data is normalized
  pub current_epoch: Cell<u64>, // for internal tracking of what the current epoch is
  pub num_samples: u64,
  pub num_train: u64,
  pub num_test: u64,
  pub num_validation: Option<u64>,
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
