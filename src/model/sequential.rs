use af;
use af::{Array, Backend, HasAfEnum};
use std::cmp::max;
use num::Zero;
use itertools::Zip;
use std::default::Default;
use std::collections::HashMap;

use loss;
use layer::{Layer, Dense, RNN, Unitary};//, LSTM};
use data::{DataSource};
use device::{Device, DeviceManager, DeviceManagerFactory};
use model::Model;
use optimizer::{Optimizer, SGD};
use params::{ParamManager, DenseGenerator, LSTMGenerator, RNNGenerator, UnitaryGenerator};

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
        self.param_manager.add_rnn::<T>(self.manager.clone(), self.device
                                        , input_size, output_size
                                        , params.get("activation").unwrap()
                                        , params.get("w_init").unwrap()
                                        , params.get("w_recurrent_init").unwrap()
                                        , params.get("b_init").unwrap());
        self.layers.push(Box::new(RNN{input_size: input_size
                                      , output_size: output_size}));
      }
      // "lstm"  => {
      //   self.param_manager.add_lstm::<T>(self.manager.clone(), self.device
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
      
      "unitary" => { 
          let hidden_size = params.get("hidden_size").unwrap().parse::<u64>().unwrap() as usize;
          self.param_manager.add_unitary::<T>(self.manager.clone(), self.device
                                            , input_size, output_size, hidden_size 

                                            // activations for Ux + Wh + b1 and for Vh + b2
                                            , params.get("h_activation").unwrap()
                                            , params.get("o_activation").unwrap()

                                            // init hidden state values
                                            , params.get("h_init").unwrap()

                                            // init values for input2hidden matrix params
                                            , params.get("v_init").unwrap()
                                            
                                            // init values for unitary matrices params
                                            , params.get("phase_init").unwrap()
                                            , params.get("householder_init").unwrap()
                                            , params.get("permut_init").unwrap()

                                            // init values for hidden2output matrix params
                                            , params.get("u_init").unwrap()
                                            
                                            // init biases values
                                            , params.get("h_bias_init").unwrap()
                                            , params.get("o_bias_init").unwrap()
                                            ); 
            self.layers.push(Box::new(Unitary{input_size: input_size
                                        , output_size: output_size}));
     }

      _  => panic!("Error unknown layer type"),
    }
  }

  //TODO: convert to log crate w/ hashmap
  fn info(&self) {
    println!("");
    self.optimizer.info();
    println!("loss:           {}\nnum_layers:     {}", self.loss, self.layers.len());
  }

  fn forward<T>(&mut self, activation: &Array
                , src_device: Device
                , dest_device: Device) -> Vec<Array>
    where T: HasAfEnum + Zero + Clone
  {
    // check & swap if the backend matches to runtime one (if not already)
    let activ = self.manager.swap_array_backend::<T>(&activation, src_device, self.device);

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

  fn fit<T, E>(&mut self, source: &T, src_device: Device
               , epochs: u64, batch_size: u64, bptt_interval: Option<u64>
               , verbose: bool) -> Vec<f32>
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

        /*
        {
            let param = self.param_manager.get_params(0);
            let ltex = param.lock().unwrap();
            af::print(&ltex.weights[0]);
            af::print(&ltex.deltas[0]);
        }
        */
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
            current_loss_vec = self.backward(&a_t, &bptt_target_slice);
          }
        }else{
          let a_t = self.forward::<E>(&batch_input, compute_device, compute_device);
          current_loss_vec = self.backward(&a_t, &batch_target);
        }

        self.optimizer.update(&mut self.param_manager, batch_size as u64);


        // cache and print loss (if verbose)
        if verbose {
          let mut total = 0f32;
          for i in &current_loss_vec{
              total += *i;
          }
          print!("{} ", total);
        }
        lossvec.extend(current_loss_vec);
      }
    }

    //utils::write_csv::<f32>("loss.csv", &lossvec);
    self.manager.swap_device(src_device); // return to src device
    lossvec
  }

  fn backward(&mut self, predictions: &Vec<Array>, targets: &Array) -> Vec<f32> {
    // setup the optimizer parameters (if not already setup)
    self.optimizer.setup(self.param_manager.get_all_dims());
    let mut loss_vec = Vec::with_capacity(predictions.len());

    for (pred, ind) in Zip::new((predictions.iter().rev(), (0..predictions.len()).rev()))
    {
      let tar = af::slice(&targets, ind as u64);
      let last_index = self.layers.len();
      let mut delta = loss::get_loss_derivative(&self.loss, pred, &tar).unwrap();
      for i in (0..last_index).rev() {
        delta = self.layers[i].backward(self.param_manager.get_params(i), &delta);
      }
      loss_vec.push(loss::get_loss(&self.loss, pred, &tar).unwrap());
    }

    loss_vec
  }
}
