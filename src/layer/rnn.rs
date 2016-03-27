use af;
use af::{Array, MatProp};

use activations;
use params::{Input, Params};
use layer::Layer;

pub struct RNN {
  pub input_size: usize,
  pub output_size: usize,
  pub bptt_interval: usize,
}

impl Layer for RNN
{
  fn forward(&self, params: &mut Params, inputs: &Input, train: bool) -> Input
  {
    // calculate our index offsets for weights
    let w_index = params.current_unroll;
    let u_index = self.bptt_interval + 1;

    // h_t = Uh_{t-1} + Wx + b [the bias is added in parallel for batch]
    let wx = af::matmul(&inputs.data            //activated_input
                        , &params.weights[w_index]
                        , MatProp::NONE
                        , MatProp::NONE).unwrap();
    let uhtm1 = af::matmul(&params.recurrences[params.current_unroll]
                           , &params.weights[u_index]
                           , MatProp::NONE
                           , MatProp::NONE).unwrap();
    let h_t = af::transpose(&af::add(&af::transpose(&uhtm1, false).unwrap(),
                                     &af::add(&af::transpose(&wx, false).unwrap()
                                              , &params.biases[0], true).unwrap(), false).unwrap()
                            , false).unwrap();

    // a_t = sigma(z_t)
    let a_t = Input{ data: activations::get_activation(&params.activations[0], &z_t).unwrap()
                     , activation: params.activations[0].clone() };

    // parameter manager keeps the outputs, inputs & recurrences
    // these are only needed for training, so dont store otherwise
    if train {
      params.inputs[params.current_unroll] = inputs.clone();
      params.outputs[params.current_unroll] = a_t.clone();
      params.recurrences[params.current_unroll] = h_t.clone();
    }

    params.current_unroll += 1;
    a_t.clone() // clone just increases the ref count
  }

  // delta_b = delta
  // delta_w = matmul(delta, activations_previous)
  fn backward(&self, params: &mut Params, delta: &Array) -> Array {
    // delta_t     = (transpose(W_{t+1}) * d_{l+1}) .* dActivation(z)
    // delta_{t-1} = (transpose(W_{t}) * d_{l})
    params.deltas = vec![af::mul(delta, &activations::get_activation_derivative(&params.activations[0]
                                                                                , &params.outputs[0].data).unwrap(), false).unwrap()];
    af::matmul(&params.deltas[0], &params.weights[0], af::MatProp::NONE, af::MatProp::TRANS).unwrap()
  }
}
