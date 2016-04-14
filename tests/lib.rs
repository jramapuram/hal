extern crate hal;
extern crate arrayfire as af;

use std::env;
use af::{Array, Dim4, Backend};

use hal::{utils, activations, initializations, loss};
use hal::layer;
use hal::layer::{Layer};
use hal::params::{DenseGenerator, ParamManager};
use hal::device::{DeviceManagerFactory, Device};
use hal::params::Input;
use hal::error::HALError;

//TODO: Move this to separate modules

/// Test derivatives
fn verify_derivative<F>(ufunc: F, name: &str, kinks: bool)
  where F : Fn(&Array) -> Array
{
  println!("\nGradient testing {}...", name);
  let dims = Dim4::new(&[32, 1, 1, 1]);
  let x = initializations::normal::<f32>(dims, 0.05f64);
  let grad = activations::get_derivative(name, &ufunc(&x)).unwrap();
  match kinks {
    true  => utils::verify_gradient_kinks(ufunc, &x, 1e-5, &grad).unwrap(),
    false => utils::verify_gradient_smooth(ufunc, &x, 1e-5, &grad).unwrap(),
  };
}

#[test]
fn lrelu_gradient() {
  verify_derivative(activations::lrelu
                    , "lrelu"
                    , true);
}

#[test]
fn tanh_gradient() {
  verify_derivative(activations::tanh
                    , "tanh"
                    , false);
}

#[test]
fn sigmoid_gradient() {
  verify_derivative(activations::sigmoid
                    , "sigmoid"
                    , false);
}

// TODO: Why is this broken? Do you need softmax + xentropy?
// #[test]
// fn softmax_gradient() {
//   verify_derivative(activations::softmax
//                     , "softmax"
//                     , false);
//}

#[test]
fn relu_gradient() {
  verify_derivative(activations::relu
                    , "relu"
                    , true);
}

#[test]
fn ones_gradient() {
  verify_derivative(activations::ones
                    , "ones"
                    , false);
}


/// Test unitary functions
fn verify_func<F>(ufunc: F, name: &str, truth: &[f32])
  where F : Fn(&Array) -> Array
{
  env::set_var("RUST_TEST_THREADS", "1");
  println!("\nTesting unitary function {}...", name);
  let dims = Dim4::new(&[5, 1, 1, 1]);
  let x = Array::new(&[-1.0, 0.0, 1.0, 2.0, 3.0], dims).unwrap();

  // verify with L2 Loss
  let x_t = Array::new(truth, dims).unwrap();
  let l2 = loss::get_loss("l2", &ufunc(&x), &x_t).unwrap();
  assert!(l2 <= 1e-4, "L2 loss of {} is higher than expected: {}", name, l2);
}


#[test]
fn tanh(){
  verify_func(activations::tanh
              , "tanh"
              , &[-0.7616, 0.0000, 0.7616, 0.9640, 0.9951]);
}

#[test]
fn sigmoid(){
  verify_func(activations::sigmoid
              , "sigmoid"
              , &[0.2689, 0.5000, 0.7311, 0.8808, 0.9526]);
}

#[test]
fn relu(){
  verify_func(activations::relu
              , "relu"
              , &[0.0, 0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn lrelu(){
  verify_func(activations::lrelu
              , "lrelu"
              , &[-0.01, 0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn softmax(){
  verify_func(activations::softmax
              , "softmax"
              , &[0.0117, 0.0317, 0.0861, 0.2341, 0.6364]);
}

#[test]
fn ones(){
  verify_func(activations::ones
              , "ones"
              , &[-1.0, 0.0, 1.0, 2.0, 3.0]);
}


/// test losses
#[test]
fn cross_entropy_softmax(){
  println!("\nTesting Cross-Entropy Soft max...");
  let dims = Dim4::new(&[5, 1, 1, 1]);
  let x = Array::new(&[-0.01, 0.00, 1.10, 2.20, 3.15], dims).unwrap();
  let target = Array::new(&[1.0, 0.00, 0.00, 0.00, 0.00], dims).unwrap();
  let x_t = 4.9980f32;

  let x_pred = loss::get_loss("cross_entropy"
                              , &activations::softmax(&x)
                              , &target).unwrap();
  let l2 = x_t * x_t - x_pred * x_pred;
  assert!(l2 < 1e-7
          , "cross-entropy-softmax loss is more than expected {} vs {} => {}"
          , x_t, x_pred, l2);
}

/// test layers
pub fn layer_helper(layer_type: &str, idims: Dim4, odims: Dim4, loss: &str
                    , eps: f64, activation: &str, w_init: &str, b_init: &str
                    , inputs: Vec<f64>, targets: Vec<f64>)
{
  println!("Testing {} Layer with activation {}...", layer_type, activation);
  env::set_var("AF_DISABLE_GRAPHICS", "1"); // GLFW crashes otherwise
  // [batch_size, input_size, temporal_size, 1]
  let input_size: usize = idims[1] as usize;
  let output_size: usize = odims[1] as usize;
  let temporal_size: usize = idims[2] as usize;

  let x = Array::new::<f64>(&inputs[..], idims).unwrap();
  let x_target_activ = Array::new::<f64>(&targets[..], odims).unwrap();

  // add a param manager, a device manager, a device
  let mut param_manager = ParamManager::default();
  let device_manager = DeviceManagerFactory::new();
  let device = Device{backend: Backend::DEFAULT, id: 0};

  // add the layer type
  let layer = match layer_type {
    "Dense" => layer::Dense {
      input_size: input_size,
      output_size: output_size,
    },
    //TODO: RNN/LSTM, etc
    _      => panic!("unknown layer type specified"),
  };

  // push it into the param manager
  match layer_type {
    "Dense" => param_manager.add_dense::<f64>(device_manager, device
                                              , input_size, output_size
                                              , activation
                                              , w_init
                                              , b_init),
  //TODO: RNN, LSTM, etc
    _      => panic!("unknown layer type specified"),
  };

  // run a forward pass and verify it
  let mut params = param_manager.get_mut_params(0);
  let activ = layer.forward(params
                            , &Input{data: x.clone(), activation: activation.to_owned()}
                            , true);
  //print!("x = "); af::print(&x.clone()).unwrap();
  //print!("x+e = "); af::print(&af::add(&x.clone(), &eps, false).unwrap()).unwrap();
  let loss_activ = loss::get_loss(loss, &activ.data, &x_target_activ).unwrap();
  assert!(loss_activ < 1e-9
          , "Forward pass verification failed, error = {}"
          , loss_activ);

  // run a forward pass on f(x + h) and f(x - h)
  let activ_p_h = layer.forward(params
                                , &Input{data: af::add(&x.clone(), &eps, false).unwrap()
                                        , activation: activation.to_owned()}
                                , false);
  let activ_m_h = layer.forward(params
                                , &Input{data: af::sub(&x.clone(), &eps, false).unwrap()
                                         , activation: activation.to_owned()}
                                , false);
  let l_p_h = loss::get_loss_vec(loss, &activ_p_h.data, &x.clone()).unwrap();
  let l_m_h = loss::get_loss_vec(loss, &activ_m_h.data, &x.clone()).unwrap();

  // get the gradient
  let delta = loss::loss_delta(&activ.data
                               , &x.copy().unwrap()
                               , loss
                               , activation);
  let grad = layer.backward(params, &delta);

  // verify gradient
  let rel = utils::gradient_check_with_perturbations(&l_p_h
                                                     , &l_m_h
                                                     , eps, &grad);
  println!("Relative error = {}", rel);
  match rel {
    n if n > 1e-2             => panic!("Gradient check failed, relative error = {}", rel),
    n if n < 1e-2 && n > 1e-4 => panic!("Gradient check failed, relative error = {}", rel),
    _                         => println!("dense test successful"),
  };
}

#[test]
fn dense_linear() {
  let idims = Dim4::new(&[1, 5, 1, 1]);
  let odims = Dim4::new(&[1, 5, 1, 1]);
  layer_helper("Dense", idims, odims, "l2", 1e-5
               , "linear"  // activation
               , "ones"    // weight init
               , "zeros"   // bias init
               , vec![-0.01, 0.00, 1.10, 2.20, 3.15] //input
               , vec![6.4400, 6.4400,6.4400, 6.4400, 6.4400]); //target
}

#[test]
fn dense_nonlinear() {
  let idims = Dim4::new(&[1, 5, 1, 1]);
  let odims = Dim4::new(&[1, 5, 1, 1]);
  layer_helper("Dense", idims, odims, "l2", 1e-5
               , "tanh"    // activation
               , "ones"    // weight init
               , "zeros"   // bias init
               , vec![-0.01, 0.00, 1.10, 2.20, 3.15] //input
               , vec![1.0000, 1.0000, 1.0000, 1.0000, 1.0000]); //target
}
