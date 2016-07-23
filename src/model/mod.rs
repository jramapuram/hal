pub use self::sequential::Sequential;
mod sequential;

use num::Zero;
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

  fn fit<T, E>(&mut self, source: &T, src_device: Device
               , epochs: u64, batch_size: u64, bptt_interval: Option<u64>
               , loss_indices: Option<&Vec<bool>>, verbose: bool) -> Vec<f32>
    where T: DataSource, E: HasAfEnum + Zero + Clone;

  fn forward<T>(&mut self, activation: &Array
                , src_device: Device
                , dest_device: Device) -> Vec<Array>
    where T: HasAfEnum + Zero + Clone;

  fn backward(&mut self, predictions: &Vec<Array>, targets: &Array, loss_indices: Option<&Vec<bool>>) -> Vec<f32>;

  fn add<T: HasAfEnum>(&mut self, layer: &str, params: HashMap<&str, String>);
  fn info(&self);
}
