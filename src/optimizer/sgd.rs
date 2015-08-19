use layer::{Layer};
use optimizer::Optimizer;
use af::{Array};
use std::collections::HashMap;
use std::default::Default;

pub struct SGD {
  name: &'static str,
  learning_rate: f32,
  momemtum: f32,
  decay: f32,
  nesterov: bool,
  clip_grad: u64,
  iter: u64,
}

impl Default for SGD {
  fn default() -> SGD { //docs say Self
    SGD {
      name: "SGD",
      learning_rate: 0.01,
      momentum: 0.0,
      decay: 0.0,
      nesterov: false,
      clip_grad: 0,
      iter: 0,
    }
  }
}

impl Optimizer for SGD { 
  fn new(params: &HashMap<&str, &str>) -> SGD {
    SGD{
      name: "SGD",
      learning_rate: params.get("learning_rate").parse::<f32>(),
      momemtum: params.get("momemtum").parse::<f32>(),
      decay: params.get("decay").parse::<f32>(),
      nesterov: params.get("nesterov").parse::<bool>(),
      clip_grad: params.get("clip_grad").parse::<u64>(),
      iter: 0,
    }
  }
  
  fn grads(&self, prediction: &Array, target: &Array
           , input: &Array, loss: &'static str) -> Array
  {
    // d_L = d_loss * d(z) where z = activation w/out non-linearity
    let d_loss = loss::get_loss_derivative(loss, prediction, target);
    let d_z = activations::get_activation_derivative(input);
    af::mul(d_loss, d_z)
  }

  //TODO: Add nesterov & momentum
  fn update_one(&self, layer: &mut Layer, diffs: &Array){
    // W = W - lr * a_{l-1} * d_l
    // b = b - lr * d_l
    layer.set_weights(af::sub(layer.get_weights()
                              , af::mul(self.learning_rate, af::mul(layer.get_inputs(), diffs))));
    layer.set_bias(layer.get_bias(), af::mul(self.learning_rate, diffs));
  }
  
  fn update(&self, layers: &mut Vec<Layer>, target: &Array, loss: &'static str){
    self.iter += 1;
    self.learning_rate *= (1.0 / (1.0 + self.decay * self.iterations));
    let mut activation = layers.last().forward(layers.last().get_inputs()); // XXX
    let mut diffs = grads(activation
                          , target
                          , layers.last().get_inputs()
                          , loss);
    self.update_one(layers.last(), diffs);
    for i in (0..layers.len() - 1).rev() {
      // d_l = (W_{l+1}^T * d_{l+1}) .* derivative(inputs)
      activation = layers[i].forward(layers[i].get_inputs())
      let grad = activations::get_activation_derivative(activation, layers[i].get_inputs());
      diffs = self.layers[i].backward(&diffs, grad);
      self.update_one(layers[i], diffs);
    }
  }
  
  fn info(&self){
    println!("name:          {}", self.name);
    println!("learning_rate: {}", self.name);
    println!("momemtum:      {}", self.name);
    println!("decay:         {}", self.name);
    println!("nesterov:      {}", self.name);
    println!("clip_grad:     {}", self.name);
  }
}
