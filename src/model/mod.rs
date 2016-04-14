pub use self::sequential::Sequential;
mod sequential;

use af::{Array, HasAfEnum};
use std::collections::HashMap;

use device::{Device, DeviceManager};
use data::{DataSource};
use optimizer::Optimizer;

pub trait Model {
  fn new(manager: DeviceManager
         , optimizer: Box<Optimizer>
         , loss: &str
         , device: Device) -> Self;

  fn fit<T: DataSource>(&mut self, source: &T, src_device: Device
         , epochs: u64, batch_size: u64, verbose: bool) -> Vec<f32>;

  fn forward(&mut self, activation: &Array, src_device: Device, dest_device: Device, train: bool) -> Array;
  fn backward(&mut self, prediction: &Array, target: &Array) -> f32;

  fn add<T: HasAfEnum>(&mut self, layer: &str, params: HashMap<&str, String>);
  fn info(&self);
}
