struct LSTM {
  input_gate : Params,
  output_gate: Params,
  forget_gate: Params,
  cell: Params,
}

impl Layer for LSTM {
  pub fn new(input_size: u64, output_size: u64) -> LSTM {
    LSTM {
      params : Params {
        weight_dims : vec![Dim4::new(&[input_size, output_size, 1, 1])],
        bias_dims : vec![Dim4::new(&[output_size, 1, 1, 1])],
        weights : vec![af::randn(Dim4::new(&[input_size, output_size, 1, 1]), af::Aftype::F32).unwrap()],
        bias : vec![af::constant(1.0 as f32, Dim4::new(&[output_size, 1, 1, 1])).unwrap()],
      },
     }
  }

  pub fn forward(&self, activation: &Array) {
    af::matmul(self.weights, activation) + self.bias
  }

  // pub fn backward(&self, inputs: &Array, gradients: &Array) {
     
  //}
}

