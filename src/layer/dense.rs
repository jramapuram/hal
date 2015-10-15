use af;
use af::{Array, MatProp};

use activations;
use params::{Input, Params};
use layer::Layer;

pub struct Dense {
  pub input_size: usize,
  pub output_size: usize,
}

impl Layer for Dense {

  fn forward(&self, params: &mut Params
             , inputs: &Input
             , recurrence: &Option<Input>) -> (Input, Option<Input>)
  {
    // keep previous_activation
    params.inputs = vec![inputs.clone()];

    // a_t = sigma(Wx + b) [the bias is added in parallel for batch]
    let a_t = af::add(&af::matmul(&params.weights[0]
                                  , &inputs.data//activated_input
                                  , MatProp::NONE
                                  , MatProp::NONE).unwrap()
                      , &params.biases[0], true).unwrap();

    (Input { data: activations::get_activation(&params.activations[0], &a_t).unwrap()
             , activation: params.activations[0].clone() }
     , None)
  }

  fn backward(&self, params: &mut Params, delta: &Array) -> Array {
    // d_lm1 = (transpose(W) * d_{l}) .* dActivation(z-1) where z = activation w/out non-linearity
    params.deltas = vec![delta.clone()];
    let activation_prev = activations::get_activation(&params.inputs[0].activation, &params.inputs[0].data).unwrap();
    let d_activation_prev = activations::get_activation_derivative(&params.inputs[0].activation, &activation_prev).unwrap();
    let delta_prev = af::mul(&af::matmul(&params.weights[0], delta, af::MatProp::TRANS, af::MatProp::NONE).unwrap()
                             , &d_activation_prev, false).unwrap();
    delta_prev
  }

}
