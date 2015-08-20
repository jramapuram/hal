use af;
use loss;
use activations;
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
      momemtum: 0.0,
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
      learning_rate: params.get("learning_rate").unwrap().parse::<f32>().unwrap(),
      momemtum: params.get("momemtum").unwrap().parse::<f32>().unwrap(),
      decay: params.get("decay").unwrap().parse::<f32>().unwrap(),
      nesterov: params.get("nesterov").unwrap().parse::<bool>().unwrap(),
      clip_grad: params.get("clip_grad").unwrap().parse::<u64>().unwrap(),
      iter: 0,
    }
  }
  
  fn grads(&self, prediction: &Array, target: &Array, input: &Array
           , loss: &'static str, activation_type: &'static str) -> Array
  {
    // d_L = d_loss * d(z) where z = activation w/out non-linearity
    let d_loss = loss::get_loss_derivative(loss, prediction, target);
    let d_z = activations::get_activation_derivative(activation_type, input);
    af::mul(d_loss, d_z).unwrap()
  }

  //TODO: Add nesterov & momentum
  fn update_one(&self, layer: &mut Box<Layer>, prev_activation: &Array, diffs: &Array){
    // W = W - lr * a_{l-1} * d_l
    let mut weights = layer.get_weights();
    for i in (0..weights.len()) {
      let update = af::mul(*prev_activation, *diffs).unwrap();
      layer.set_weights(af::sub(weights[i], af::mul(self.learning_rate, update).unwrap()).unwrap(), i);
    }
    // b = b - lr * d_l
    let mut biases = layer.get_bias();
    for i in (0..biases.len()) {
      layer.set_bias(af::sub(biases[i], af::mul(self.learning_rate, *diffs).unwrap()).unwrap(), i);
    }
  }
  
  fn update(&self, layers: &mut Vec<Box<Layer>>
            , prediction: &Array
            , target: &Array
            , loss: &'static str) -> (Array, Array)
  {
    self.iter += 1;
    self.learning_rate *= (1.0 / (1.0 + self.decay * (self.iter as f32)));
    let &(mut prev_activation, mut current_wxb) = layers.last().unwrap().get_inputs();
    let mut diffs = self.grads(prediction
                               , target
                               , &current_wxb
                               , loss
                               , layers.last().unwrap().get_activation_type());
    self.update_one(&mut layers.last().unwrap(), &prev_activation, &diffs);

    // Verify:: Don't backprop to 1st layer
    for i in (1..layers.len() - 1).rev() {
      // d_l = (W_{l+1}^T * d_{l+1}) .* derivative(z) where z = activation w/out non-linearity
      let &(mut prev_activation, mut current_wxb) = layers[i].get_inputs();
      let grad = activations::get_activation_derivative(layers[i].get_activation_type(), &current_wxb);
      diffs = layers[i].backward(&diffs, &grad);
      self.update_one(&mut layers[i], &prev_activation, &diffs);
    }

    (loss::get_loss(loss, prediction, target), *prediction)
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
