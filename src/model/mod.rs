//mod Optimizer;
//mod Loss;

pub use self::sequential::Sequential;
mod sequential;

use af::{Array, Backend};
use std::collections::HashMap;

use device::{Device, DeviceManager};
use optimizer::Optimizer;

pub trait Model {
  fn new(manager: DeviceManager
         , optimizer: Box<Optimizer>
         , loss: &str
         , device: Device) -> Self;
  fn fit(&mut self, input: &mut Array, target: &mut Array, batch_size: u64
         , shuffle: bool, verbose: bool) -> Vec<f32>;
  fn forward(&mut self, activation: &Array, train: bool) -> Array;
  fn backward(&mut self, prediction: &Array, target: &Array) -> f32;
  fn add(&mut self, layer: &str
         , params: HashMap<&str, String>);
  fn info(&self);
}
