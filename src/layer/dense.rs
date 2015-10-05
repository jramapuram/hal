use af;
use af::{Dim4, Array, MatProp};

use activations;
use initializations;
use params::{Input, Params};
use layer::Layer;

pub struct Dense {
  input_size: usize,
  output_size: usize,
}

// impl Dense {
//   fn new(param_manager: &ParamManager
//          , input_size: u64, output_size: u64
//          , activation: &'static str
//          , w_init: &'static str
//          , b_init: &'static str) -> Dense
//   {
//     param_manager.add_dense(input_size
//                             , output_size
//                             , activation
//                             , w_init
//                             , b_init);
//     Dense{
//       name: "dense",
//       input_size: input_size,
//       output_size: output_size,
//     }
//   }
// }

impl Layer for Dense {

  fn forward(&self, params: &mut Params
             , inputs: &Input
             , recurrence: &Input) -> (Input, Input)
  {
    // keep previous_activation
    params.inputs = vec![inputs.clone()];

    // apply the activation to the previous layer [Optimization: Memory saving]
    let activated_input = activations::get_activation(inputs.activation
                                                      , &inputs.data).unwrap();

    // sigma(Wx + b)
    let mul = af::matmul(&params.weights[0]
                         , &activated_input
                         , MatProp::NONE
                         , MatProp::NONE).unwrap();
    (Input { data: af::add(&mul, &params.biases[0], true).unwrap()
             , activation: params.activations[0] }
     , recurrence.clone())
  }

  fn backward(&self, params: &mut Params, delta: &Array) -> Array {
    // d_l = (transpose(W) * d_{l}) .* dActivation(z-1) where z = activation w/out non-linearity
    params.delta = vec![delta.clone()];
    let activation_prev = activations::get_activation(params.inputs[0].activation, &params.inputs[0].data).unwrap();
    let d_activation_prev = activations::get_activation_derivative(params.inputs[0].activation, &activation_prev).unwrap();
    let delta_prev = af::mul(&af::matmul(&params.weights[0], delta, af::MatProp::TRANS, af::MatProp::NONE).unwrap()
                             , &d_activation_prev, false).unwrap();
    delta_prev
  }

}
