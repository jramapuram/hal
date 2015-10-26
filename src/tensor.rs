use af;
use af::{Dim4, Array, AfBackend};
use std::fmt;
use std::cell::Cell;
use std::ops::{Add, Sub, Mul, Div};

use device::{Device, DeviceManager};

pub struct Tensor<'a> {
  array: Array,
  device: Device,
  manager: &'a DeviceManager,
}

impl<'a> Tensor<'a> {
  fn new(array: Array, manager: &DeviceManager
         , backend: AfBackend, device: i32) -> Tensor {
    Tensor { manager: manager
             , array: array
             , device: Device{ backend: backend, id: device } }
  }

  fn get(&self) -> &Array {
    self.manager.swap_device(self.device);
    &self.array
  }

  fn get_mut(&mut self) -> &mut Array {
    self.manager.swap_device(self.device);
    &mut self.array
  }

  fn set(&mut self, update: &Array) {
    self.manager.swap_device(self.device);
    self.array.clone_from(update);
  }

  fn batch_add(&self, other: &Tensor, batch: bool) -> Tensor<'a> {
    if (other.device != self.device) {
      panic!("add: can't mix between two different devices");
    }

    Tensor { array: af::add(&self.array, &other.array, batch).unwrap()
             , device: self.device
             , manager: self.manager }
  }

  fn batch_sub(&self, other: &Tensor, batch: bool) -> Tensor<'a> {
    if (other.device != self.device) {
      panic!("sub: can't mix between two different devices");
    }

    Tensor { array: af::sub(&self.array, &other.array, batch).unwrap()
             , device: self.device
             , manager: self.manager }
  }

  fn batch_mul(&self, other: &Tensor, batch: bool) -> Tensor {
    if (other.device != self.device) {
      panic!("mul: can't mix between two different devices");
    }

    Tensor { array: af::mul(&self.array, &other.array, batch).unwrap()
             , device: self.device
             , manager: self.manager }
  }

  fn batch_div(&self, other: &Tensor, batch: bool) -> Tensor {
    if (other.device != self.device) {
      panic!("div: can't mix between two different devices");
    }

    Tensor { array: af::div(&self.array, &other.array, batch).unwrap()
             , device: self.device
             , manager: self.manager }
  }
}

impl<'a> fmt::Debug for Tensor<'a> {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "{:?}\nbackend: {:?}, device: {}", self.array.dims().unwrap().get().clone()
           , self.device.backend, self.device.id)
  }
}

impl<'a> Clone for Tensor<'a> {
    fn clone(&self) -> Tensor<'a> {
      self.manager.swap_device(self.device);
      Tensor{
        array: self.array.clone(),
        device: self.device.clone(),
        manager: self.manager,
      }
    }
}

macro_rules! get_mut_param_vec_func {
  ($fn_name: ident, $vec_extension: ident, $base_type: ty) => (
    #[allow(unused_mut)]
    pub fn $fn_name(&mut self, layer_index: usize) -> &mut Vec<$base_type> {
      assert!(self.layer_storage.len() - 1 >= layer_index);
      &mut self.layer_storage[layer_index].$vec_extension
    }
    )
}

macro_rules! scalar_impl (
    ($operand: ident, $fn_name: ident, $foo:ty) => (
      impl $operand<foo> for Tensor<'a> {
        type Output = Tensor;
        fn $fn_name(self, rhs: $foo) -> Tensor<'a> {
          Tensor { array: af::$fn_name(&self.array, &rhs, false).unwrap()
                   , device: self.device
                   , manager: self.manager }
        }
      }
      impl $fn_name<Tensor> for $foo {
        type Output = Tensor;
        fn $fn_name(self, rhs : Tensor) -> Tensor<'a> {
          Tensor { array: af::$fn_name(&self.array, &rhs, false).unwrap()
                   , device: self.device
                   , manager: self.manager }
        }
      }
      )
    );

impl Add for Tensor {
  type Output = Tensor;

  fn add(self, other: Tensor) -> Tensor {

  }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        Tensor {x: self.x - other.x, y: self.y - other.y}
    }
}
