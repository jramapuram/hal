use af;
use af::{Array, MatProp};

use activations;
use params::{Input, Params};
use layer::Layer;

pub struct Dense {
  pub input_size: usize,
  pub output_size: usize,
}

impl Layer for Dense
{
  fn forward(&self, params: &mut Params, inputs: &Input, train: bool)-> Input
  {
    // z_t = Wx + b [the bias is added in parallel for batch]
    let wx = af::matmul(&inputs.data            //activated_input
                        , &params.weights[0]
                        , MatProp::NONE
                        , MatProp::NONE).unwrap();
    let z_t = af::transpose(&af::add(&af::transpose(&wx, false).unwrap()
                                     , &params.biases[0], true).unwrap(), false).unwrap();

    // a_t = sigma(z_t)
    let a_t = Input{ data: activations::get_activation(&params.activations[0], &z_t).unwrap()
                     , activation: params.activations[0].clone() };

    // parameter manager keeps the output & inputs
    // these are only needed for training, so dont store otherwise
    if train {
      params.inputs = vec![inputs.clone()];
      params.outputs = vec![a_t.clone()];
    }

    a_t.clone() // clone just increases the ref count
  }

  fn backward(&self, params: &mut Params, delta: &Array) -> Array {
    // delta_t     = (transpose(W_{t+1}) * d_{l+1}) .* dActivation(z)
    // delta_{t-1} = (transpose(W_{t}) * d_{l})
    let dz = activations::get_derivative(&params.activations[0]
                                         , &params.outputs[0].data).unwrap();
    let delta_t = af::mul(delta, &dz, false).unwrap();
    let dw = af::matmul(&params.inputs[0].data, &delta_t // delta_w = delta_t * a_{t-1}
                        , af::MatProp::TRANS
                        , af::MatProp::NONE).unwrap();
    let db = af::transpose(&af::sum(&delta_t, 0).unwrap(), false).unwrap(); // delta_b = delta
    params.deltas[0] = af::add(&params.deltas[0], &dw, false).unwrap();
    params.deltas[1] = af::add(&params.deltas[1], &db, false).unwrap();

    af::matmul(&delta_t, &params.weights[0], af::MatProp::NONE, af::MatProp::TRANS).unwrap()
  }
}
