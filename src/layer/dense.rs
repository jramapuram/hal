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

  fn forward(&self, params: &mut Params, inputs: &Input)-> Input
  {
    // parameter manager keeps previous activation
    params.inputs = vec![inputs.clone()];

    // a_t = sigma(Wx + b) [the bias is added in parallel for batch]
    let a_t = af::add(&af::matmul(&params.weights[0]
                                  , &inputs.data//activated_input
                                  , MatProp::NONE
                                  , MatProp::NONE).unwrap()
                      , &params.biases[0], true).unwrap();

    // parameter manager keeps the output
    params.outputs = vec![activations::get_activation(&params.activations[0], &a_t).unwrap()];
    params.outputs[0].clone() //clone is merely incrementing a refcount
  }

  fn backward(&self, params: &mut Params, delta: &Array) -> Array {
    // delta_t     = (transpose(W_{t+1}) * d_{l+1}) .* dActivation(z)
    // delta_{t-1} = (transpose(W_{t}) * d_{l})
    params.deltas = vec![af::mul(delta, &activations::get_activation_derivative(params.activations[0], params.outputs[0]).unwrap()).unwrap()];
    af::matmul(&params.weights[0], &params.deltas[0], af::MatProp::TRANS, af::MatProp::NONE).unwrap()
  }

}
