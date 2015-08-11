use layer::{Layer};
use af::{Array};
use std::collections::HashMap
use std::default::Default
  
#[derive(Default)]
pub struct SGD {
  name: &'static str,
  learning_rate: f32,
  momemtum: f32,
  decay: f32,
  nesterov: bool,
  clip_grad: bool,
}

// def get_updates(self, params, constraints, loss):
//         grads = self.get_gradients(loss, params)
//         lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
//         self.updates = [(self.iterations, self.iterations + 1.)]

//         for p, g, c in zip(params, grads, constraints):
//             m = shared_zeros(p.get_value().shape)  # momentum
//             v = self.momentum * m - lr * g  # velocity
//             self.updates.append((m, v))

//             if self.nesterov:
//                 new_p = p + self.momentum * v - lr * g
//             else:
//                 new_p = p + v

//             self.updates.append((p, c(new_p)))  # apply constraints
//         return self.updates

impl Optimizer, Default for SGD {
  fn default() -> SGD { //docs say Self
    SGD{
      name: "SGD",
      learning_rate: 0.01,
      momentum: 0.0,
      decay: 0.0,
      nesterov: false,
      clip_grad: false,
    }
  }
  
  fn new(params: &HashMap) -> SGD {
    SGD{
      name: "SGD",
      learning_rate: params.get("learning_rate"),
      momemtum: params.get("momemtum"),
      decay: params.get("decay"),
      nesterov: params.get("nesterov"),
      clip_grad: params.get("clip_grad"),
    }
  }
  
  fn grads(&self, layer: &Layer) -> Array{
    
  }
  
  fn update(&mut self, layer: &mut Layer, grads: &Array){
    self.learning_rate *= (1.0 / (1.0 + self.decay * self.iterations));
    // COMPLETE THIS
  }
  
  fn info(){
    println!("name:          {}", self.name);
    println!("learning_rate: {}", self.name);
    println!("momemtum:      {}", self.name);
    println!("decay:         {}", self.name);
    println!("nesterov:      {}", self.name);
    println!("clip_grad:     {}", self.name);
  }
}
