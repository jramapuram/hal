#[macro_use] extern crate hal;
extern crate arrayfire as af;

use std::cell::{RefCell, Cell};

use hal::Model;
use hal::optimizer::{Optimizer, SGD, get_optimizer_with_defaults};
use hal::data::{Data, DataSource, DataParams, Normalize, Shuffle};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::plot::{plot_vec, plot_array};
use hal::device::{DeviceManagerFactory, Device};
use af::{Array, Dim4, Aftype, Backend};


/********* Build a custom data source that returns [x, y] tuples *******/
pub struct SinSource {
  pub params: DataParams,
  pub iter: Cell<u64>,
  pub offset: Cell<f32>,
}

impl SinSource {
  fn new(input_size: u64, batch_size: u64
         , max_samples: u64, is_normalized: bool
         , is_shuffled: bool) -> SinSource
  {
    let dims = Dim4::new(&[batch_size, input_size, 1, 1]);
    let train_samples = 0.7 * max_samples as f32;
    let test_samples = 0.2 * max_samples as f32;
    let validation_samples = 0.1 * max_samples as f32;
    SinSource {
      params: DataParams {
        input_dims: dims,   // input is the same size as the output
        target_dims: dims,  // ^
        normalize: is_normalized,
        shuffle: is_shuffled,
        current_epoch: Cell::new(0),
        num_samples: max_samples,
        num_train: train_samples as u64,
        num_test: test_samples as u64,
        num_validation: Some(validation_samples as u64),
      },
      iter: Cell::new(0),
      offset: Cell::new(0.0f32),
    }
  }

  fn generate_sin_wave(&self, input_dims: u64, num_rows: u64) -> Array {
    let tdims = Dim4::new(&[input_dims, num_rows, 1, 1]);
    let dims = Dim4::new(&[1, num_rows * input_dims, 1, 1]);
    let x = af::transpose(&af::moddims(&af::range(dims, 1, Aftype::F32).unwrap()
                                       , tdims).unwrap(), false).unwrap();
    let x_shifted = af::add(&self.offset.get()
                            , &af::div(&x, &(input_dims*num_rows), false).unwrap()
                            , true).unwrap();
    self.offset.set(self.offset.get() + 1.0/input_dims as f32);
    af::sin(&x_shifted).unwrap()
  }
}

impl DataSource for SinSource
{
  fn get_train_iter(&self, num_batch: u64) -> Data {
    let inp = self.generate_sin_wave(self.params.input_dims[1], num_batch);
    let mut batch = Data {
      input: RefCell::new(Box::new(inp.clone())),
      target: RefCell::new(Box::new(inp.copy().unwrap())),
    };

    if self.params.normalize { batch.normalize(3.0); }
    if self.params.shuffle   {  batch.shuffle(); }
    let current_iter = self.params.current_epoch.get();
    if self.iter.get()  == self.params.num_samples as u64/ num_batch as u64 {
      self.params.current_epoch.set(current_iter + 1);
    }
    self.iter.set(self.iter.get() + 1);
    batch
  }

  fn info(&self) -> DataParams {
    self.params.clone()
  }

  fn get_test_iter(&self, num_batch: u64) -> Data {
    self.get_train_iter(num_batch)
  }

  fn get_validation_iter(&self, num_batch: u64) -> Option<Data> {
    Some(self.get_train_iter(num_batch))
  }
}
/****************** End Data Source Definition ********************/

fn main() {
  // First we need to parameterize our network
  let input_dims = 64;
  let hidden_dims = 32;
  let output_dims = 64;
  let num_train_samples = 65536;
  let batch_size = 128;
  let optimizer_type = "SGD";
  let epochs = 5;

  // Now, let's build a model with an device manager on a specific device
  // an optimizer and a loss function. For this example we demonstrate a simple autoencoder
  // AF_BACKEND_DEFAULT is: OpenCL -> CUDA -> CPU
  let manager = DeviceManagerFactory::new();
  let gpu_device = Device{backend: Backend::AF_BACKEND_DEFAULT, id: 0};
  let cpu_device = Device{backend: Backend::AF_BACKEND_CPU, id: 0};
  let optimizer = get_optimizer_with_defaults(optimizer_type).unwrap();
  let mut model = Box::new(Sequential::new(manager.clone()
                                           , optimizer         // optimizer
                                           , "mse"             // loss
                                           , gpu_device));     // device for model

  // Let's add a few layers why don't we?
  model.add("dense", hashmap!["activation"    => "tanh".to_string()
                              , "input_size"  => input_dims.to_string()
                              , "output_size" => hidden_dims.to_string()
                              , "w_init"      => "glorot_uniform".to_string()
                              , "b_init"      => "zeros".to_string()]);
  model.add("dense", hashmap!["activation"    => "tanh".to_string()
                              , "input_size"  => hidden_dims.to_string()
                              , "output_size" => output_dims.to_string()
                              , "w_init"      => "glorot_uniform".to_string()
                              , "b_init"      => "zeros".to_string()]);

  // Get some nice information about our model
  model.info();

  // Temporarily set the backend to CPU so that we can load data into RAM
  // The model will automatically toggle to the desired backend during training
  manager.swap_device(cpu_device);

  // Build our sin wave source
  let sin_generator = SinSource::new(input_dims, batch_size
                                     , num_train_samples
                                     , false   // normalized
                                     , false); // shuffled

  // Pull a sample to verify sizing
  let test_sample = sin_generator.get_train_iter(batch_size);
  println!("test sample shape: {:?}"
           , test_sample.input.into_inner().dims().unwrap());

  // iterate our model in Verbose mode (printing loss)
  // Note: more manual control can be enacted by directly calling
  //       forward/backward & optimizer update
  let loss = model.fit(&sin_generator        // what data source to pull from
                       , cpu_device          // source device
                       , epochs, batch_size  // self explanatory :)
                       , true);              // verbose

  // plot our loss on a 512x512 grid with the provided title
  plot_vec(loss, "Loss vs. Iterations", 512, 512);

  // infer on one test and plot the first sample (row) of the predictions
  let test_sample = sin_generator.get_test_iter(1).input.into_inner();
  let prediction = model.forward(&test_sample
                                 , cpu_device // source device
                                 , cpu_device // destination device
                                 , false);    // not training
  println!("\nprediction shape: {:?} | backend = {:?}"
           , prediction.dims().unwrap(), prediction.get_backend());
  plot_array(&af::flat(&prediction).unwrap(), "Model Inference", 512, 512);
}
