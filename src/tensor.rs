use af;
use af::{Array, AfBackend, MatProp};
use std::fmt;
use std::sync::{Arc, RwLock};
use std::ops::{Add, Sub, Mul, Div, Deref, DerefMut};

use device::{Device, DeviceManager};

pub struct Tensor {
  array: Array,
  device: Device,
  manager: Arc<RwLock<DeviceManager>>,
}

impl Tensor {
  fn new(array: Array, manager: &Arc<RwLock<DeviceManager>>
         , backend: AfBackend, device: i32) -> Tensor {
    Tensor { manager: manager.clone()
             , array: array
             , device: Device{ backend: backend, id: device } }
  }

  fn get(&self) -> &Array {
    let m = self.manager.read().unwrap();
    m.swap_device(self.device);
    &self.array
  }

  fn get_mut(&mut self) -> &mut Array {
    let m = self.manager.read().unwrap();
    m.swap_device(self.device);
    &mut self.array
  }

  fn set(&mut self, update: &Array) {
    let m = self.manager.read().unwrap();
    m.swap_device(self.device);
    self.array.clone_from(update);
  }

  fn batch_add(&self, other: &Tensor, batch: bool) -> Tensor {
    if (other.device != self.device) {
      panic!("add: can't mix between two different devices");
    }

    let m = self.manager.read().unwrap();
    m.swap_device(self.device);

    Tensor { array: af::add(&self.array, &other.array, batch).unwrap()
             , manager: self.manager.clone()
             , device: self.device }
  }

  fn batch_sub(&self, other: &Tensor, batch: bool) -> Tensor {
    if other.device != self.device {
      panic!("sub: can't mix between two different devices");
    }

    let m = self.manager.read().unwrap();
    m.swap_device(self.device);

    Tensor { array: af::sub(&self.array, &other.array, batch).unwrap()
             , manager: self.manager.clone()
             , device: self.device }
  }

  fn batch_mul(&self, other: &Tensor, batch: bool) -> Tensor {
    if other.device != self.device {
      panic!("mul: can't mix between two different devices");
    }

    let m = self.manager.read().unwrap();
    m.swap_device(self.device);

    Tensor { array: af::mul(&self.array, &other.array, batch).unwrap()
             , manager: self.manager.clone()
             , device: self.device }
  }

  fn batch_div(&self, other: &Tensor, batch: bool) -> Tensor {
    if other.device != self.device {
      panic!("div: can't mix between two different devices");
    }

    let m = self.manager.read().unwrap();
    m.swap_device(self.device);

    Tensor { array: af::div(&self.array, &other.array, batch).unwrap()
             , manager: self.manager.clone()
             , device: self.device }
  }

  fn matmul(&self, other: &Tensor, lhs_prop: MatProp, rhs_prop: MatProp) -> Tensor
  {
    if other.device != self.device {
      panic!("div: can't mix between two different devices");
    }

    let m = self.manager.read().unwrap();
    m.swap_device(self.device);

    Tensor { array: af::matmul(&self.array, &other.array, lhs_prop, rhs_prop).unwrap()
             , manager: self.manager.clone()
             , device: self.device }
  }
}

impl Deref for Tensor {
  type Target = Array;
   fn deref<'a>(&'a self) -> &'a Array {
     self.get()
   }
}

impl DerefMut for Tensor {
  fn deref_mut<'a>(&'a mut self) -> &'a mut Array {
    self.get_mut()
  }
}

impl fmt::Debug for Tensor {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "{:?}\nbackend: {:?}, device: {}", self.array.dims().unwrap().get().clone()
           , self.device.backend, self.device.id)
  }
}

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
    let m = self.manager.read().unwrap();
    m.swap_device(self.device);

      Tensor{
        array: self.array.clone(),
        device: self.device.clone(),
        manager: self.manager.clone(),
      }
    }
}

impl Add<Tensor> for Tensor {
  type Output = Tensor;
  fn add(self, other: Tensor) -> Tensor {
    self.batch_add(&other, false)
  }
}

impl Sub<Tensor> for Tensor {
  type Output = Tensor;
  fn sub(self, other: Tensor) -> Tensor {
    self.batch_sub(&other, false)
  }
}

impl Mul<Tensor> for Tensor {
  type Output = Tensor;
  fn mul(self, other: Tensor) -> Tensor {
    self.batch_mul(&other, false)
  }
}

impl Div<Tensor> for Tensor {
  type Output = Tensor;
  fn div(self, other: Tensor) -> Tensor {
    self.batch_div(&other, false)
  }
}

// TODO: Enable f64/u64, etc support separately
macro_rules! algebra_impl (
  ($operand: ident, $fn_name: ident, $foo: ty) => (
    impl $operand<$foo> for Tensor {
      type Output = Tensor;
      fn $fn_name(self, rhs: $foo) -> Tensor {
        let rhs_float = rhs as f32;
        Tensor { array: af::$fn_name(&self.array, &rhs_float, false).unwrap()
                 , device: self.device
                 , manager: self.manager.clone() }
      }
    }
    impl $operand<Tensor> for $foo {
      type Output = Tensor;
      fn $fn_name(self, rhs : Tensor) -> Tensor {
        let lhs_float = self as f32;
        Tensor { array: af::$fn_name(&lhs_float, &rhs.array, false).unwrap()
                 , device: rhs.device
                 , manager: rhs.manager.clone() }
      }
    }
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
