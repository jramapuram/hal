use af;
use af::{Array, Backend, HasAfEnum};
use std::cmp::max;
use num::Zero;
use itertools::Zip;
use std::default::Default;
use std::collections::HashMap;

use loss;
use utils;
use layer::{Layer, Dense, RNN};//, LSTM};
use data::{DataSource};
use device::{Device, DeviceManager, DeviceManagerFactory};
use model::Model;
use optimizer::{Optimizer, SGD};
use params::{ParamManager, DenseGenerator, LSTMGenerator, RNNGenerator};

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
      device: Device{ backend: Backend::DEFAULT, id: 0 },
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

  /// Adds a new layer to the sequential model
  ///
  /// Given a layer type and provided parameters this function
  /// will add the required parameters to the sequential model
  ///
  /// # Parameters
  ///
  /// - `layer` is the type of layer to add
  /// - `params` is a hashmap of params for the provided layer
  fn add<T: HasAfEnum>(&mut self, layer: &str
                       , params: HashMap<&str, String>)
  {
    //TODO: Error handling for hashmap
    let input_size = params.get("input_size").unwrap().parse::<u64>().unwrap() as usize;
    let output_size = params.get("output_size").unwrap().parse::<u64>().unwrap() as usize;
    match layer {
      "dense" => {
        self.param_manager.add_dense::<T>(self.manager.clone(), self.device
                                          , input_size, output_size
                                          , params.get("activation").unwrap()
                                          , params.get("w_init").unwrap()
                                          , params.get("b_init").unwrap());
        self.layers.push(Box::new(Dense{input_size: input_size
                                        , output_size: output_size}));
      },
      "rnn" => {
        let hidden_size = params.get("hidden_size").unwrap().parse::<u64>().unwrap() as usize;
        self.param_manager.add_rnn::<T>(self.manager.clone(), self.device
                                        , input_size, hidden_size, output_size
                                        , params.get("inner_activation").unwrap()
                                        , params.get("outer_activation").unwrap()
                                        , params.get("w_init").unwrap()
                                        , params.get("b_init").unwrap());
        self.layers.push(Box::new(RNN{input_size: input_size
                                      , hidden_size: hidden_size
                                      , output_size: output_size}));
      }
      // "lstm"  => {
      //   self.param_manager.add_lstm::<T>(self.manager.clone(), self.device
      //                               , input_size, output_size
      //                               , params.get("input_activation").unwrap()
      //                               , params.get("outer_activation").unwrap()
      //                               , params.get("w_init").unwrap()
      //                               , params.get("w_recurrent_init").unwrap()
      //                               , params.get("forget_b_init").unwrap()
      //                               , params.get("b_init").unwrap());
      //   self.layers.push(Box::new(LSTM{input_size: input_size
      //                                  , output_size: output_size}));
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

  /// Calculate the forward pass of all the layers
  ///
  /// Given an array of inputs this function computes the forward pass
  /// on all the available layers and return the final model outputs
  ///
  /// # Parameters
  ///
  /// - `inputs` is an array of activations [batch, feature, time]
  /// - `src_device` is the source device that the data is coming from
  /// - `dest_device` is the destination device that the data should go to
  ///
  /// # Return Values
  ///
  /// Vector of activated outputs of the model
  fn forward<T>(&mut self, inputs: &Array
                , src_device: Device
                , dest_device: Device) -> Vec<Array>
    where T: HasAfEnum + Zero + Clone
  {
    // check & swap if the backend matches to runtime one (if not already)
    let activ = self.manager.swap_array_backend::<T>(&inputs, src_device, self.device);

    // if dim[3] > 1 we assume we have an RNN
    // we will need to unwind at least once for non RNNs
    let bptt_unroll = max(activ.dims()[2], 1);
    let mut activate;

    for t in 0..bptt_unroll {
      activate = af::slice(&activ, t);
      for i in 0..self.layers.len() {
        let (a, _) = self.layers[i].forward(self.param_manager.get_params(i)
                                            , &activate, None);
        activate = a;
      }
    }

    // TODO: Parameterize
    // zero the states
    // self.param_manager.zero_all_states(None);

    // return the collected outputs of the last layer
    let last_index = self.layers.len() - 1;
    let mut outputs = self.param_manager.get_outputs(last_index);

    // return to the dest device
    for i in 0..outputs.len() {
      outputs[i] = self.manager.swap_array_backend::<T>(&outputs[i], self.device, dest_device);
    }

    outputs
  }

  /// Fit's model to provided data
  ///
  /// Given input and output data, fits the model the given data by running
  /// both forward and backward pass on the data and optimizing the system per minibatch
  ///
  /// # Parameters
  ///
  /// - `source` is the datasource
  /// - `src_device` is the source device of the data
  /// - `epochs` is the number of epochs to run the training loop for
  /// - `batch_size` is the minibatch size
  /// - `bptt_interval` is the optional parameter for truncated backprop through time (RNN's only)
  /// - `loss_indices` are the indices to utilize when doing backward pass (useful for RNN long term tasks)
  /// - `verbose` specifies whether or not to print verbose details during training
  ///
  /// # Return Values
  ///
  /// Vector of losses
  fn fit<T, E>(&mut self, source: &T, src_device: Device
               , epochs: u64, batch_size: u64, bptt_interval: Option<u64>
               , loss_indices: Option<&Vec<bool>>, verbose: bool) -> Vec<f32>
    where T: DataSource, E: HasAfEnum + Zero + Clone
  {
    // some simple data validity checks
    let data_params = source.info();
    let idims = data_params.input_dims;
    let tdims = data_params.target_dims;
    let iters =  data_params.num_samples as u64 / batch_size as u64;
    println!("\ntrain samples: {:?} | target samples: {:?} | batch size: {}"
             , idims, tdims, batch_size);
    println!("epochs: {} | iterations[per epoch]: {}", epochs, iters);
    assert!(idims[0] == tdims[0]
            , "batch sizes for inputs and targets much be equal");
    assert!(idims[2] == tdims[2]
            , "sequence lengths for inputs and targets must be equal");
    assert!(self.layers.len() > 0
            , "Need at least one layer to fit!");

    // verify that last layer is of logits type when using
    // softmax_crossentropy or binary_crossentropy
    if self.loss.to_lowercase() == "cross_entropy_softmax"
      || self.loss.to_lowercase() == "binary_cross_entropy"
    {
      let last_layer_index = self.layers.len() - 1;
      let last_layer_activations = self.param_manager.get_activations(last_layer_index);
      let last_activation = last_layer_activations.last().unwrap();
      assert!(last_activation == "ones" || last_activation == "linear",
              "Erroneous results expected while using cross_entropy_* \
               loss and non-logit units in the last layer: {}"
              , last_activation);
    }


    // loss vector current loss
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
        assert!(minibatch.input.borrow().dims()[0] == batch_size
                , "Ensure that input dims are of batch rows");
        assert!(minibatch.target.borrow().dims()[0] == batch_size
                , "Ensure that target dims are of batch rows");
        let batch_input = self.manager.swap_array_backend::<E>(&minibatch.input.into_inner()
                                                          , src_device
                                                          , compute_device);
        let batch_target = self.manager.swap_array_backend::<E>(&minibatch.target.into_inner()
                                                           , src_device
                                                           , compute_device);

        // if bptt_interval is specified we slice our minibatch
        // into bptt_interval number of slices and then forward pass on it
        let mut current_loss_vec = Vec::new();
        if let Some(bptt_interval) = bptt_interval {
          let num_seqs = idims[2]/bptt_interval;
          let start: Vec<_>  = (0..num_seqs).map(|x| x * bptt_interval).collect();
          let finish: Vec<_> = (1..num_seqs+1).map(|x| x * bptt_interval).collect();
          for (begin, end) in Zip::new((start, finish)) //TODO: fix when .step_by() becomes stable
          {
            let bptt_input_slice = af::slices(&batch_input, begin, end-1);
            let bptt_target_slice = af::slices(&batch_target, begin, end-1);
            let a_t = self.forward::<E>(&bptt_input_slice, compute_device, compute_device);
            current_loss_vec = self.backward(&a_t, &bptt_target_slice, loss_indices);
          }
        }else{
          let a_t = self.forward::<E>(&batch_input, compute_device, compute_device);
          current_loss_vec = self.backward(&a_t, &batch_target, loss_indices);
        }

        self.optimizer.update(&mut self.param_manager, batch_size as u64);

        // cache and print loss (if verbose)
        if verbose {
          let loss_sum = current_loss_vec.iter().fold(0f32, |sum, val| sum + val);
          let avg_loss = current_loss_vec.len() as f32 * loss_sum;
          print!("{} ", avg_loss);
        }
        lossvec.extend(current_loss_vec);
      }
    }

    //utils::write_csv::<f32>("loss.csv", &lossvec);
    self.manager.swap_device(src_device); // return to src device
    lossvec
  }


  /// Calculate the layer gradients and return the loss vector
  ///
  /// Given predictions and output data, this function computes all the gradients for all
  /// of the trainable parameters in the layers
  ///
  /// # Parameters
  ///
  /// - `predictions` are the model predictions
  /// - `targets` are the true targets
  /// - `loss_indices` are the optional indices of losses to use while computing the gradient
  ///
  /// # Return Values
  ///
  /// Vector of losses
  fn backward(&mut self, predictions: &Vec<Array>, targets: &Array, loss_indices: Option<&Vec<bool>>) -> Vec<f32> {
    // setup the optimizer parameters (if not already setup)
    self.optimizer.setup(self.param_manager.get_all_dims());
    let mut loss_vec = Vec::with_capacity(predictions.len());

    for (pred, ind) in Zip::new((predictions.iter().rev(), (0..predictions.len()).rev()))
    {
      let tar = af::slice(&targets, ind as u64);
      let last_index = self.layers.len();

      // handle loss indices that are not to be allowed
      let mut delta = match loss_indices {
        Some(li) => {
          assert!(li.len() == predictions.len()
                  , "loss indices need to be of the same size as the predictions");
          match li[ind] {
            false => utils::constant(tar.dims(), tar.get_type(), 0.0f32),
            true  => {
              loss_vec.push(loss::get_loss(&self.loss, pred, &tar).unwrap());
              loss::get_loss_derivative(&self.loss, pred, &tar).unwrap()
            },
          }
        },
        None     => {
          loss_vec.push(loss::get_loss(&self.loss, pred, &tar).unwrap());
          loss::get_loss_derivative(&self.loss, pred, &tar).unwrap()
        },
      };

      for i in (0..last_index).rev() {
        delta = self.layers[i].backward(self.param_manager.get_params(i), &delta);
      }
    }

    loss_vec
  }
}
