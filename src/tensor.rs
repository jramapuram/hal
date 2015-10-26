use af;
use af::{Dim4, Array, AfBackend, MatProp};
use std::fmt;
use std::cell::RefCell;
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

  fn batch_mul(&self, other: &Tensor, batch: bool) -> Tensor<'a> {
    if (other.device != self.device) {
      panic!("mul: can't mix between two different devices");
    }

    Tensor { array: af::mul(&self.array, &other.array, batch).unwrap()
             , device: self.device
             , manager: self.manager }
  }

  fn batch_div(&self, other: &Tensor, batch: bool) -> Tensor<'a> {
    if (other.device != self.device) {
      panic!("div: can't mix between two different devices");
    }

    Tensor { array: af::div(&self.array, &other.array, batch).unwrap()
             , device: self.device
             , manager: self.manager }
  }

  fn matmul(&self, other: &Tensor, lhs_prop: MatProp, rhs_prop: MatProp) -> Tensor<'a>
  {
    if (other.device != self.device) {
      panic!("div: can't mix between two different devices");
    }

    Tensor { array: af::matmul(&self.array, &other.array, lhs_prop, rhs_prop).unwrap()
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

impl<'a> Add<Tensor<'a>> for Tensor<'a> {
  type Output = Tensor<'a>;
  fn add(self, other: Tensor) -> Tensor<'a> {
    self.batch_add(&other, false)
  }
}

impl<'a> Sub<Tensor<'a>> for Tensor<'a> {
  type Output = Tensor<'a>;
  fn sub(self, other: Tensor) -> Tensor<'a> {
    self.batch_sub(&other, false)
  }
}

impl<'a> Mul<Tensor<'a>> for Tensor<'a> {
  type Output = Tensor<'a>;
  fn mul(self, other: Tensor) -> Tensor<'a> {
    self.batch_mul(&other, false)
  }
}

impl<'a> Div<Tensor<'a>> for Tensor<'a> {
  type Output = Tensor<'a>;
  fn div(self, other: Tensor) -> Tensor<'a> {
    self.batch_div(&other, false)
  }
}

// TODO: Enable f64/u64, etc support separately
macro_rules! algebra_impl (
  ($operand: ident, $fn_name: ident, $foo: ty) => (
    impl<'a> $operand<$foo> for Tensor<'a> {
      type Output = Tensor<'a>;
      fn $fn_name(self, rhs: $foo) -> Tensor<'a> {
        let rhs_float = rhs as f32;
        Tensor { array: af::$fn_name(&self.array, &rhs_float, false).unwrap()
                 , device: self.device
                 , manager: self.manager }
      }
    }
    // impl<'a> $operand<Tensor<'a>> for $foo {
    //   type Output = Tensor<'a>;
    //   fn $fn_name(self, rhs : Tensor) -> Tensor<'a> {
    //     let lhs_float = self as f32;
    //     Tensor { array: af::$fn_name(&lhs_float, &rhs.array, false).unwrap()
    //              , device: rhs.device
    //              , manager: rhs.manager }
    //   }
    // }
    ));

algebra_impl!(Mul, mul, i8);
algebra_impl!(Mul, mul, i16);
algebra_impl!(Mul, mul, i32);
algebra_impl!(Mul, mul, i64);
algebra_impl!(Mul, mul, isize);
algebra_impl!(Mul, mul, usize);
algebra_impl!(Mul, mul, u8);
algebra_impl!(Mul, mul, u16);
algebra_impl!(Mul, mul, u32);
algebra_impl!(Mul, mul, u64);
algebra_impl!(Mul, mul, f32);
algebra_impl!(Mul, mul, f64);

algebra_impl!(Add, add, i8);
algebra_impl!(Add, add, i16);
algebra_impl!(Add, add, i32);
algebra_impl!(Add, add, i64);
algebra_impl!(Add, add, isize);
algebra_impl!(Add, add, usize);
algebra_impl!(Add, add, u8);
algebra_impl!(Add, add, u16);
algebra_impl!(Add, add, u32);
algebra_impl!(Add, add, u64);
algebra_impl!(Add, add, f32);
algebra_impl!(Add, add, f64);

algebra_impl!(Sub, sub, i8);
algebra_impl!(Sub, sub, i16);
algebra_impl!(Sub, sub, i32);
algebra_impl!(Sub, sub, i64);
algebra_impl!(Sub, sub, isize);
algebra_impl!(Sub, sub, usize);
algebra_impl!(Sub, sub, u8);
algebra_impl!(Sub, sub, u16);
algebra_impl!(Sub, sub, u32);
algebra_impl!(Sub, sub, u64);
algebra_impl!(Sub, sub, f32);
algebra_impl!(Sub, sub, f64);

algebra_impl!(Div, div, i8);
algebra_impl!(Div, div, i16);
algebra_impl!(Div, div, i32);
algebra_impl!(Div, div, i64);
algebra_impl!(Div, div, isize);
algebra_impl!(Div, div, usize);
algebra_impl!(Div, div, u8);
algebra_impl!(Div, div, u16);
algebra_impl!(Div, div, u32);
algebra_impl!(Div, div, u64);
algebra_impl!(Div, div, f32);
algebra_impl!(Div, div, f64);
