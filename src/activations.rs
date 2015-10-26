use af;
use af::Array;
use error::HALError;

pub fn tanh(x: &Array) -> Array {
  af::tanh(x).unwrap()
}

pub fn sigmoid(x: &Array) -> Array {
  // let exponentiated = x.map(|e| 1.0/(1.0 + af::exp(-1.0 * e)));
  // exponentiated.unwrap()
  let denominator = af::add(&1.0f32, &af::exp(&af::mul(&-1.0f32, x, false).unwrap()).unwrap(), false).unwrap();
  af::div(&1.0f32, &denominator, false).unwrap()
}

pub fn softmax(x: &Array) -> Array {
  let exponentiated = af::exp(&x).unwrap();
  // let exponentialted_sum = af::sum(exponentiated).unwrap();
  // let smax = exponentiated.map(|elem| af::div(elem, exponentialted_sum)).unwrap();
  // smax
  af::div(&exponentiated, &af::sum_all(&exponentiated).unwrap().0, false).unwrap()
}

pub fn tanh_derivative(x: &Array) -> Array {
  // 1 - tanh(x)*tanh(x)
  // let t = tanh(x);
  // af::sub(&1.0f32, &af::mul(&t, &t, false).unwrap(), false).unwrap()
  af::sub(&1.0f32, &af::mul(x, x, false).unwrap(), false).unwrap()
}

pub fn sigmoid_derivative(x: &Array) -> Array {
  // x * (1 - x)
  //let s = sigmoid(x);
  //af::mul(&s, &af::sub(&1.0f32, &s, false).unwrap(), false).unwrap()
  af::mul(x, &af::sub(&1.0f32, x, false).unwrap(), false).unwrap()
}

pub fn softmax_derivative(x: &Array) -> Array {
  // x * (1 - x)
  //let s = softmax(x);
  //af::mul(&s, &af::sub(&1.0f32, &s, false).unwrap(), false).unwrap()
  sigmoid_derivative(x)
}

pub fn ones(x: &Array) -> Array {
  x.clone()
}


pub fn get_activation(name: &str, x: &Array) -> Result<Array, HALError> {
  match name {
    "softmax" => Ok(softmax(x)),
    "sigmoid" => Ok(sigmoid(x)),
    "tanh"    => Ok(tanh(x)),
    "ones"    => Ok(ones(x)),
    _         => Err(HALError::UNKNOWN),
  }
}

pub fn get_activation_derivative(name: &str, x: &Array) -> Result<Array, HALError> {
  match name {
    "softmax" => Ok(softmax_derivative(x)),
    "sigmoid" => Ok(sigmoid_derivative(x)),
    "tanh"    => Ok(tanh_derivative(x)),
    "ones"    => Ok(ones(x)),
    _         => Err(HALError::UNKNOWN),
  }
}

// use af;

// use tensor::Tensor;
// use error::HALError;

//   // fn new(array: Array, manager: &DeviceManager
//   //        , backend: AfBackend, device: i32) -> Tensor {

// pub fn tanh(x: &Tensor) -> Tensor {
//   x.clone().set(af::tanh(x.get()).unwrap())
// }

// pub fn sigmoid(x: &Tensor) -> Tensor {
//   // let exponentiated = x.map(|e| 1.0/(1.0 + af::exp(-1.0 * e)));
//   // exponentiated.unwrap()
//   let denominator = af::add(&1.0f32, &af::exp(&af::mul(&-1.0f32, x.get(), false).unwrap()).unwrap(), false).unwrap();
//   x.clone().set(af::div(&1.0f32, &denominator, false).unwrap())
// }

// pub fn softmax(x: &Tensor) -> Tensor {
//   let exponentiated = af::exp(&x.get()).unwrap();
//   // let exponentialted_sum = af::sum(exponentiated).unwrap();
//   // let smax = exponentiated.map(|elem| af::div(elem, exponentialted_sum)).unwrap();
//   // smax
//   x.clone().set(af::div(&exponentiated, &af::sum_all(&exponentiated).unwrap().0, false).unwrap())
// }

// pub fn tanh_derivative(x: &Tensor) -> Tensor {
//   // 1 - tanh(x)*tanh(x)
//   // let t = tanh(x);
//   // af::sub(&1.0f32, &af::mul(&t, &t, false).unwrap(), false).unwrap()
//   x.clone().set(af::sub(&1.0f32, &af::mul(x.get(), x.get(), false).unwrap(), false).unwrap())
// }

// pub fn sigmoid_derivative(x: &Tensor) -> Tensor {
//   // x * (1 - x)
//   //let s = sigmoid(x);
//   //af::mul(&s, &af::sub(&1.0f32, &s, false).unwrap(), false).unwrap()
//   af::mul(x.get(), &af::sub(&1.0f32, x.get(), false).unwrap(), false).unwrap()
// }

// pub fn softmax_derivative(x: &Tensor) -> Tensor {
//   // x * (1 - x)
//   //let s = softmax(x);
//   //af::mul(&s, &af::sub(&1.0f32, &s, false).unwrap(), false).unwrap()
//   sigmoid_derivative(x.get())
// }

// pub fn ones(x: &Tensor) -> Tensor {
//   x.clone()
// }


// pub fn get_activation(name: &str, x: &Tensor) -> Result<Tensor, HALError> {
//   match name {
//     "softmax" => Ok(softmax(x)),
//     "sigmoid" => Ok(sigmoid(x)),
//     "tanh"    => Ok(tanh(x)),
//     "ones"    => Ok(ones(x)),
//     _         => Err(HALError::UNKNOWN),
//   }
// }

// pub fn get_activation_derivative(name: &str, x: &Tensor) -> Result<Tensor, HALError> {
//   match name {
//     "softmax" => Ok(softmax_derivative(x)),
//     "sigmoid" => Ok(sigmoid_derivative(x)),
//     "tanh"    => Ok(tanh_derivative(x)),
//     "ones"    => Ok(ones(x)),
//     _         => Err(HALError::UNKNOWN),
//   }
// }
