use af;
use af::{Array, Backend};
use std::cmp::max;
use std::default::Default;
use std::collections::HashMap;

use utils;
use loss;
use layer::{Layer, Dense};//, LSTM};
use data::{DataSource};
use device::{Device, DeviceManager, DeviceManagerFactory};
use model::Model;
use optimizer::{Optimizer, SGD};
use params::{ParamManager, DenseGenerator, LSTMGenerator, Input};

pub struct Sequential {
  layers: Vec<Box<Layer>>,
  param_manager: ParamManager,
  optimizer: Box<Optimizer>,
  manager: DeviceManager,
  loss: String,
  device: Device,
}

impl Default for Sequential {
  fn default() -> Sequential {
    Sequential {
      layers: Vec::new(),
      param_manager: ParamManager::default(),
      optimizer: Box::new(SGD::default()),
      manager: DeviceManagerFactory::new(),
      loss: "mse".to_string(),
      device: Device{ backend: Backend::AF_BACKEND_DEFAULT, id: 0 },
    }
  }
}

impl Drop for Sequential {
  fn drop(&mut self) {
    self.manager.swap_device(self.device);
  }
}

impl Model for Sequential {
  fn new(manager: DeviceManager
         , optimizer: Box<Optimizer>
         , loss: &str
         , device: Device) -> Sequential {
    Sequential {
      layers: Vec::new(),
      param_manager: ParamManager::default(),
      manager: manager,
      loss: loss.to_string(),
      optimizer: optimizer,
      device: device,
    }
  }

  fn add(&mut self, layer: &str
         , params: HashMap<&str, String>)
  {
    //TODO: Error handling for hashmap
    let input_size = params.get("input_size").unwrap().parse::<u64>().unwrap() as usize;
    let output_size = params.get("output_size").unwrap().parse::<u64>().unwrap() as usize;
    match layer {
      "dense" => {
        self.param_manager.add_dense(self.manager.clone(), self.device
                                     , input_size, output_size
                                     , params.get("activation").unwrap()
                                     , params.get("w_init").unwrap()
                                     , params.get("b_init").unwrap());
        self.layers.push(Box::new(Dense{input_size: input_size
                                        , output_size: output_size}));
      },
      // "lstm"  => {
      //   self.param_manager.add_lstm(self.manager.clone(), self.device
      //                               , input_size, output_size
      //                               , params.get("max_seq_size").unwrap()
      //                               , params.get("input_activation").unwrap()
      //                               , params.get("outer_activation").unwrap()
      //                               , params.get("w_init").unwrap()
      //                               , params.get("w_recurrent_init").unwrap()
      //                               , params.get("forget_b_init").unwrap()
      //                               , params.get("b_init").unwrap());
      //   self.layers.push(Box::new(LSTM{input_size: input_size
      //                                  , output_size: output_size
      //                                  , max_seq_size: params.get("max_seq_size").unwrap()
      //                                  , return_sequences: params.get("return_sequences").unwrap()}));
      // },
      _  => panic!("Error unknown layer type"),
    }
  }

  //TODO: convert to log crate w/ hashmap
  fn info(&self) {
    println!("");
    self.optimizer.info();
    println!("loss:           {}\nnum_layers:     {}", self.loss, self.layers.len());
  }

  fn forward(&mut self, activation: &Array
             , src_device: Device
             , dest_device: Device
             , train: bool) -> Array {
    // check & swap if the backend matches to runtime one (if not already)
    let activ = self.manager.swap_array_backend(&activation, src_device, self.device);

    // if dim[3] > 1 we assume we have an RNN
    // we will need to unwind at least once for non RNNs
    let bptt_unroll = max(activ.dims().unwrap()[2], 1);
    let mut activate = Input {data: af::slice(&activ, 0).unwrap()
                              , activation: "ones".to_string()};
    for t in 0..bptt_unroll {
      activate.data = af::slice(&activ, t).unwrap();
      for i in 0..self.layers.len() {
        activate = self.layers[i].forward(self.param_manager.get_mut_params(i)
                                          , &activate, train);
      }
    }

    // return to the dest device
    self.manager.swap_array_backend(&activate.data, self.device, dest_device)
  }

  fn fit<T: DataSource>(&mut self, source: &T, src_device: Device
         , epochs: u64, batch_size: u64, verbose: bool) -> Vec<f32>
  {
    // some simple data validity checks
    let data_params = source.info();
    let idims = data_params.input_dims;
    let tdims = data_params.target_dims;
    let iters =  data_params.num_samples as u64 / batch_size as u64;
    println!("\ntrain samples: {:?} | target samples: {:?} | batch size: {}"
             , idims, tdims, batch_size);
    println!("epochs: {} | iterations[per epoch]: {}", epochs, iters);
    assert!(tdims[0] == idims[0]);          // batches are of equal size

    // loss vector current loss
    let mut loss: f32;
    let mut lossvec = Vec::<f32>::new();
    let compute_device = self.device.clone();

    // iterate epoch times over the number of batch iterations
    for epoch in 0..epochs {
      for iter in 0..iters {
        // ensure we are on the original device device
        self.manager.swap_device(src_device);

        if verbose {
          print!("\n[epoch: {}][iter: {}] ", epoch, iter);
        }

        // extract part of the array onto the GPU
        self.manager.swap_device(src_device);
        let minibatch = source.get_train_iter(batch_size);
        assert!(minibatch.input.borrow().dims().unwrap()[0] == batch_size
                , "Ensure that input dims are of batch rows");
        assert!(minibatch.target.borrow().dims().unwrap()[0] == batch_size
                , "Ensure that target dims are of batch rows");
        let batch_input = self.manager.swap_array_backend(&minibatch.input.into_inner()
                                                          , src_device
                                                          , compute_device);
        let batch_target = self.manager.swap_array_backend(&minibatch.target.into_inner()
                                                           , src_device
                                                           , compute_device);

        let a_t = self.forward(&batch_input, compute_device, compute_device, true);
        loss = self.backward(&a_t, &batch_target);
        self.optimizer.update(&mut self.param_manager, batch_size as u64);
        lossvec.push(loss);

        if verbose {
          print!("{} ", loss);
        }
      }
    }

    utils::write_csv::<f32>("loss.csv", &lossvec);
    self.manager.swap_device(src_device); // return to src device
    lossvec
  }

  fn backward(&mut self, prediction: &Array, target: &Array) -> f32 {
    // setup the optimizer parameters (if not already setup)
    self.optimizer.setup(self.param_manager.get_all_dims());

    // Note: a requirement here is that the output activation is
    // the last element of the activation's vector of the last layer
    let last_index = self.layers.len() - 1;
    let last_layer_activations = self.param_manager.get_activations(last_index);
    let final_activation_index = last_layer_activations.len();
    let mut delta = loss::loss_delta(prediction
                                     , target
                                     , &self.loss
                                     , &last_layer_activations[final_activation_index - 1]);
    for i in (0..last_index + 1).rev() {
      delta = self.layers[i].backward(self.param_manager.get_mut_params(i), &delta);
    }

    loss::get_loss(&self.loss, prediction, target).unwrap()
  }
}
