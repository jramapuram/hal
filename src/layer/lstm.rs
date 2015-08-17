use af;
use af::{Dim4, Array};
use af::MatProp;
use activations;
use initializations;
use layer::{Weights, Bias, Layer};

struct LSTM {
  Wi: Weights,
  Wo: Weights,
  Wf: Weights,
  Wc: Weights,
  bi: Bias,
  bo: Bias,
  bf: Bias,
  bc: Bias,
  iDiffs: Bias,
  oDiffs: Bias,
  fDiffs: Bias,
  cDiffs: Bias,
  inner: &str,
  outer: &str,
}

//TODO: Fix the sizes for the weights/biases
impl RecurrentLayer for LSTM {
  pub fn new(input_size: u64, output_size: u64
             , inner_activation: &str, outer_activation: &str
             , w_init: &str, b_init: &str) -> LSTM {
    LSTM {
      Wi : Weights {
        dims : vec![Dim4::new(&[input_size, output_size, 1, 1])],
        weights : vec![get_initialization(w_init, Dim4::new(&[output_size, input_size, 1, 1]))],
      },
      bi: Bias {
        dims : vec![Dim4::new(&[output_size, 1, 1, 1])],
        bias : vec![get_initialization(b_init, Dim4::new(&[output_size, 1, 1, 1]))],
      },
      Wo : Weights {
        dims : vec![Dim4::new(&[input_size, output_size, 1, 1])],
        weights : vec![get_initialization(w_init, Dim4::new(&[output_size, input_size, 1, 1]))],
      },
      bo: Bias {
        dims : vec![Dim4::new(&[output_size, 1, 1, 1])],
        bias : vec![get_initialization(b_init, Dim4::new(&[output_size, 1, 1, 1]))],
      },
      Wf : Weights {
        dims : vec![Dim4::new(&[input_size, output_size, 1, 1])],
        weights : vec![get_initialization(w_init, Dim4::new(&[output_size, input_size, 1, 1]))],
      },
      bf: Bias {
        dims : vec![Dim4::new(&[output_size, 1, 1, 1])],
        bias : vec![get_initialization(b_init, Dim4::new(&[output_size, 1, 1, 1]))],
      },
      Wc : Weights {
        dims : vec![Dim4::new(&[input_size, output_size, 1, 1])],
        weights : vec![get_initialization(w_init, Dim4::new(&[output_size, input_size, 1, 1]))],
      },
      bc: Bias {
        dims : vec![Dim4::new(&[output_size, 1, 1, 1])],
        bias : vec![get_initialization(b_init, Dim4::new(&[output_size, 1, 1, 1]))],
      },
      iDiffs: zeros(Dim4::new(&[output_size, 1, 1, 1])),
      iDiffs: zeros(Dim4::new(&[output_size, 1, 1, 1])),
      iDiffs: zeros(Dim4::new(&[output_size, 1, 1, 1])),
      iDiffs: zeros(Dim4::new(&[output_size, 1, 1, 1])),
      inner: inner_activation,
      outer: outer_activation
    }
  }

  pub fn forward(&self, activation: &Array) {
    af::matmul(self.weights, activation) + self.bias
  }

  // pub fn backward(&self, inputs: &Array, gradients: &Array) {
     
  //}
}

