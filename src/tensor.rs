use af;
use af::{Dim4, Array, AfBackend};
use std::fmt;

use utils::set_device;

#[derive(Clone)]
pub struct Tensor {
  array: Array,
  backend: AfBackend,
  device: i32,
}

impl Tensor {
  fn new(array: Array, backend: AfBackend, device: i32) -> Tensor {
    Tensor { array: array, backend: backend, device: device }
  }

  fn get(&self) -> &Array {
    set_device(self.backend, self.device);
    &self.array
  }

  fn get_mut(&mut self) -> &mut Array {
    set_device(self.backend, self.device);
    &mut self.array
  }

  fn clone(&self) -> Array {
    set_device(self.backend, self.device);
    self.array.clone()
  }

  fn set(&mut self, update: &Array) {
    set_device(self.backend, self.device);
    self.array.clone_from(update);
  }
}

impl fmt::Debug for Tensor {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "{:?}\nbackend: {:?}, device: {}", self.array.dims().unwrap().get().clone()
           , self.backend, self.device)
  }
}
