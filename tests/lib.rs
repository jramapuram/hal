extern crate hal;
extern crate arrayfire as af;
extern crate itertools;
extern crate rand;
#[macro_use] extern crate timeit;

use std::env;
use af::{Array, Dim4, Backend, DType};
use itertools::Zip;
use rand::distributions::{IndependentSample, Range};

use hal::{utils, activations, initializations, loss};
use hal::layer;
use hal::layer::{Layer};
use hal::params::{DenseGenerator, RNNGenerator, ParamManager};
use hal::device::{DeviceManagerFactory, Device};
use hal::error::HALError;


//todo: move all these tests into separate modules


/// test derivatives
fn verify_derivative<F>(ufunc: F, name: &str)
  where F : Fn(&Array) -> Array
{
  println!("\ngradient testing {}...", name);
  let dims = Dim4::new(&[1, 1, 1, 1]);

  // generate a random number between (-1, 1)
  let between = Range::new(-1f32, 1f32); // utils::constant needs an f32
  let mut rng = rand::thread_rng();
  let rnd_num = between.ind_sample(&mut rng);

  // build a constant r^1 array
  let x = utils::constant(dims, DType::F64, rnd_num);

  // get the original activation and the symbolic gradient
  let activated = ufunc(&x);
  let grad = activations::get_derivative(name, &activated).unwrap();

  // run the algorithm on non-smooth function based this is a single func
  // [ie non-chained] and thus should be almost exact
  utils::verify_gradient_kinks(|i| {
    let rhs = ufunc(&i);
    let v = utils::array_to_vec(&rhs);
    v[0]
  }, &x, 1e-5, &grad).unwrap();
}

#[test]
fn lrelu_gradient() {
  verify_derivative(activations::lrelu
                    , "lrelu");
}

#[test]
fn tanh_gradient() {
  verify_derivative(activations::tanh
                    , "tanh");
}

#[test]
fn sigmoid_gradient() {
  verify_derivative(activations::sigmoid
                    , "sigmoid");
}

// todo: need to implement softmax as jacobian
// #[test]
// fn softmax_gradient() {
//   verify_derivative(activations::softmax
//                     , "softmax"
//                     , false);
//}

#[test]
fn relu_gradient() {
  verify_derivative(activations::relu
                    , "relu");
}

#[test]
fn ones_gradient() {
  verify_derivative(activations::ones
                    , "ones");
}


/// test unitary functions
fn verify_func<F>(ufunc: F, name: &str, input: &[f32], truth: &[f32])
  where F : Fn(&Array) -> Array
{
  env::set_var("rust_test_threads", "1");
  println!("\ntesting unitary function {}...", name);
  let ilen = input.len();
  let tlen = truth.len();
  assert!(ilen == tlen, "input and output lengths must be the same");

  let dims = Dim4::new(&[1, ilen as u64, 1, 1]);
  let x = Array::new::<f32>(input, dims);

  // verify with l2 loss
  let x_t = Array::new::<f32>(truth, dims);
  let l2 = loss::get_loss("l2", &ufunc(&x), &x_t).unwrap();
  assert!(l2 <= 1e-4, "l2 loss of {} is higher than expected: {}", name, l2);
}

#[test]
fn tanh(){
  verify_func(activations::tanh
              , "tanh"
              , &[-1.0, 0.0, 1.0, 2.0, 3.0]
              , &[-0.7616, 0.0000, 0.7616, 0.9640, 0.9951]);
}

#[test]
fn sigmoid(){
  verify_func(activations::sigmoid
              , "sigmoid"
              , &[-1.0, 0.0, 1.0, 2.0, 3.0]
              , &[0.2689, 0.5000, 0.7311, 0.8808, 0.9526]);
}

#[test]
fn relu(){
  verify_func(activations::relu
              , "relu"
              , &[-1.0, 0.0, 1.0, 2.0, 3.0]
              , &[0.0, 0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn lrelu(){
  verify_func(activations::lrelu
              , "lrelu"
              , &[-1.0, 0.0, 1.0, 2.0, 3.0]
              , &[-0.01, 0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn softmax(){
  verify_func(activations::softmax
              , "softmax"
              , &[-1.0, 0.0, 1.0, 2.0, 3.0]
              , &[0.01165623, 0.03168492, 0.08612854, 0.23412165, 0.63640863]);
}

#[test]
fn ones(){
  verify_func(activations::ones
              , "ones"
              , &[-1.0, 0.0, 1.0, 2.0, 3.0]
              , &[-1.0, 0.0, 1.0, 2.0, 3.0]);
}


///
/// test losses
///

/// Loss test helper function
fn verify_loss_func(name: &str, input: &[f32], target: &[f32], l2: f32)
{
  env::set_var("rust_test_threads", "1");
  println!("\ntesting loss function {}...", name);
  let ilen = input.len();
  let tlen = target.len();
  assert!(ilen == tlen, "input and output lengths must be the same");

  let dims = Dim4::new(&[1, ilen as u64, 1, 1]);
  let x = Array::new::<f32>(input, dims);
  let target = Array::new::<f32>(target, dims);
  let pred_loss = loss::get_loss(name, &x, &target).unwrap();

  // verify with l2 loss
  let true_l2 = l2 * l2 - pred_loss * pred_loss;
  assert!(true_l2 <= 1e-4
          , "l2 loss of {} is higher than expected: {}[observed] vs {}[provided] : diff {}"
          , name, pred_loss, l2, true_l2);
}

#[test]
fn cross_entropy_softmax(){
  verify_loss_func("cross_entropy_softmax"
                   , &[-0.01, 0.00, 1.10, 2.20, 3.15]
                   , &[1.0, 0.00, 0.00, 0.00, 0.00]
                   , 3.6304748f32);
}

#[test]
fn binary_cross_entropy(){
  verify_loss_func("binary_cross_entropy"
                   , &[-2.3]
                   , &[1.0]
                   , 2.3955455);
}


/// helper to build a layer
pub fn layer_builder<F>(layer_type: &str, idims: Dim4, odims: Dim4, loss: &str
                        , eps: f64, activation: &str, w_init: &str, b_init: &str, mut f: F)
  where F: FnMut(&ParamManager, Box<Layer>)
{
  // [batch_size, input_size, temporal_size, 1]
  let batch_size: usize = idims[0] as usize;
  let input_size: usize = idims[1] as usize;
  let output_size: usize = odims[1] as usize;
  let temporal_size: usize = idims[2] as usize;

  // add a param manager, a device manager, a device
  let mut param_manager = ParamManager::default();
  let device_manager = DeviceManagerFactory::new();
  let device = Device{backend: Backend::DEFAULT, id: 0};

  // add the layer type
  let layer: Box<Layer> = match layer_type {
    "Dense" => Box::new(layer::Dense {
      input_size: input_size,
      output_size: output_size,
    }),
    "rnn"  => Box::new(layer::RNN {
      input_size: input_size,
      output_size: output_size,
    }),
    //todo: lstm, etc
    _      => panic!("unknown layer type specified"),
  };

  // push it into the param manager
  match layer_type {
    "Dense" => param_manager.add_dense::<f64>(device_manager, device
                                              , input_size, output_size
                                              , activation
                                              , w_init
                                              , b_init),
    "rnn"  => {
      param_manager.add_rnn::<f64>(device_manager, device
                                   , input_size, output_size
                                   , activation
                                   , w_init, w_init // just dupe it
                                   , b_init);
    }
    //todo: lstm, etc
    _      => panic!("unknown layer type specified"),
  };

  // run the closure
  f(&mut param_manager, layer);
}

/// test forward pass for layers
pub fn layer_forward_helper(layer_type: &str, idims: Dim4, odims: Dim4, loss: &str
                            , eps: f64, activation: &str, w_init: &str, b_init: &str
                            , inputs_vec: Vec<f64>, targets_vec: Vec<f64>)
{
  //env::set_var("af_disable_graphics", "1"); // glfw crashes otherwise
  println!("testing {} layer with {} acivation for forward pass..."
           , layer_type, activation);
  let x = Array::new::<f64>(&inputs_vec[..], idims);
  let targets = Array::new::<f64>(&targets_vec[..], odims);

  layer_builder(layer_type, idims, odims, loss
                , eps, activation, w_init, b_init, |param_manager, layer|
                {
                  // run a forward pass and verify it is within tolerance
                  let params = param_manager.get_params(0);

                  // make it such that we are within an unrolling [for rnn types]
                  let hdims = Dim4::new(&[idims[0] as u64, odims[1] as u64, 1, 1]);
                  let h_t = utils::constant(hdims, DType::F64, 0.5f32);

                  // run a forward pass
                  let (activ, _) = layer.forward(params.clone(), &x.clone(), Some(h_t));

                  let loss_activ = loss::get_loss(loss, &activ, &targets).unwrap();
                  assert!(loss_activ < 1e-9
                          , "forward pass verification failed, error = {}"
                          , loss_activ);
                });
}

/// test layers gradients
pub fn layer_backward_helper(layer_type: &str, idims: Dim4, odims: Dim4, loss: &str
                             , eps: f64, activation: &str, w_init: &str, b_init: &str)
{
  //env::set_var("af_disable_graphics", "1"); // glfw crashes otherwise
  // [batch_size, input_size, temporal_size, 1]
  let input_size: usize = idims[1] as usize;
  let output_size: usize = odims[1] as usize;
  let temporal_size: usize = idims[2] as usize;

  let x = initializations::uniform::<f64>(idims, -0.5f32, 0.5f32);
  let targets = match (loss == "cross_entropy_softmax" || activation == "softmax")
  {
    true => {
      // randomly pick one of K indexes to set to 1
      let mut v: Vec<f64> = vec![0f64; output_size];
      let between = Range::new(0usize, output_size as usize);
      let mut rng = rand::thread_rng();
      let rnd_index = between.ind_sample(&mut rng);
      v[rnd_index] = 1f64;

      // build an array
      utils::vec_to_array::<f64>(v, odims)
    },
    _  => initializations::uniform::<f64>(odims, -0.5f32, 0.5f32),
  };

  layer_builder(layer_type, idims, odims, loss
                , eps, activation, w_init, b_init, |param_manager, layer|
                {
                  // run a forward and then bkwd pass to extract the gradients
                  let params = param_manager.get_params(0);

                  // make it such that we are within an unrolling [for rnn types]
                  let hdims = Dim4::new(&[idims[0] as u64, odims[1] as u64, 1, 1]);
                  let h_t = utils::constant(hdims, DType::F64, 0.5f32);

                  // run a forward pass
                  let (activ, _) = layer.forward(params.clone(), &x.clone(), Some(h_t.clone()));

                  // get the derivative and save away all params
                  let delta = loss::get_loss_derivative(loss, &activ, &targets).unwrap();
                  layer.backward(params.clone(), &delta);
                  let grads = param_manager.get_all_deltas();
                  let num_params = param_manager.num_arrays(0);

                  // iterate over all arrays and grads and run gradient checking
                  for (arr, grad, ind) in Zip::new((param_manager.get_all_arrays().iter() // weights + biases
                                                    , grads                               // tabulated gradients
                                                    , 0..num_params))                     // param index iterator
                  {
                    let arr_bkp: Array = arr.copy(); // keep a backup
                    println!("\nTesting gradient of array with {:?} dims", arr.dims());

                    // do the gradient check specific to the activation type
                    match activations::is_smooth(activation) {
                      false => utils::verify_gradient_kinks(|i| {
                        // run forward pass using the modified array
                        let p = params.clone();
                        p.lock().unwrap().current_unroll = 0;
                        param_manager.set_array_from_index(i.clone(), ind);
                        let (fwd_pass, _) = layer.forward(params.clone(), &x.clone(), Some(h_t.clone()));
                        loss::get_loss(loss, &fwd_pass, &targets).unwrap() as f64
                      }, &arr_bkp, eps, &grad).unwrap(),
                      true  => utils::verify_gradient_smooth(|i| {
                        // run forward pass using the modified array
                        let p = params.clone();
                        p.lock().unwrap().current_unroll = 0;
                        param_manager.set_array_from_index(i.clone(), ind);
                        let (fwd_pass, _) = layer.forward(params.clone(), &x.clone(), Some(h_t.clone()));
                        loss::get_loss(loss, &fwd_pass, &targets).unwrap() as f64
                      }, &arr_bkp, eps, &grad).unwrap(),
                    };
                  }
                });
}

#[test]
fn dense_forward(){
  timeit!({
    let idims = Dim4::new(&[1, 5, 1, 1]);
    let odims = Dim4::new(&[1, 5, 1, 1]);
    layer_forward_helper("Dense", idims, odims, "l2", 1e-4
                         , "linear"                                      // activation
                         , "ones"                                        // weight init
                         , "zeros"                                       // bias init
                         , vec![-0.01, 0.00, 1.10, 2.20, 3.15]           //input
                         , vec![6.4400, 6.4400,6.4400, 6.4400, 6.4400]); //target
  });
}

#[test]
fn dense_backward() {
  timeit! ({
    let idims = Dim4::new(&[1, 5, 1, 1]);
    let odims = Dim4::new(&[1, 5, 1, 1]);
    layer_backward_helper("Dense", idims, odims
                          , "l2"              // loss
                          , 1e-4              // eps for numerical grad
                          , "tanh"            // activation
                          , "glorot_uniform"  // weight init
                          , "zeros");         // bias init
  });
}

#[test]
fn rnn_forward(){
  timeit!({
    let idims = Dim4::new(&[1, 5, 1, 1]);
    let odims = Dim4::new(&[1, 5, 1, 1]);
    layer_forward_helper("rnn", idims, odims, "l2", 1e-4
                         , "linear"                                      // activation
                         , "ones"                                        // weight init
                         , "zeros"                                       // bias init
                         , vec![-0.01, 0.00, 1.10, 2.20, 3.15]           //input
                         , vec![ 8.94000053, 8.94000053, 8.94000053, 8.94000053, 8.94000053]); //target
  });
}

#[test]
fn rnn_backward() {
  timeit! ({
    let idims = Dim4::new(&[1, 5, 1, 1]); // single time slice
    let odims = Dim4::new(&[1, 6, 1, 1]); // single time slice
    layer_backward_helper("rnn", idims, odims
                          , "l2"              // loss
                          , 1e-4              // eps for numerical grad
                          , "tanh"            // activation
                          , "glorot_uniform"  // weight init
                          , "zeros");         // bias init
  });
}
