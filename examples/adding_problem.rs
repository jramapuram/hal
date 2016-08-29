#[macro_use] extern crate hal;
extern crate arrayfire as af;
extern crate rand;
use rand::distributions::{Range, IndependentSample};

use hal::utils;
use hal::Model;
use hal::loss;
use hal::initializations::uniform;
use hal::optimizer::{Optimizer, get_optimizer_with_defaults};
use hal::data::{DataSource, AddingProblemSource, CopyingProblemSource};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::plot::{plot_vec, plot_array};
use hal::device::{DeviceManagerFactory, Device};
use af::{Backend, HasAfEnum, MatProp, DType, Dim4};


fn main() {
    // First we need to parameterize our network
    let hidden_dims = 100;
    let num_train_samples = 1000000;
    let batch_size = 5;
    let optimizer_type = "Adam";
    let epochs = 1;
    let bptt_unroll: u64 = 6;

    let mut loss_indices = vec!(false; (bptt_unroll-1) as usize);
    loss_indices.push(true);

    let num_test_samples = 1000;


    // Now, let's build a model with an device manager on a specific device
    // an optimizer and a loss function. For this example we demonstrate a simple autoencoder
    // AF_BACKEND_DEFAULT is: OpenCL -> CUDA -> CPU
    let manager = DeviceManagerFactory::new();
    let gpu_device = Device{backend: Backend::DEFAULT, id: 0};
    let cpu_device = Device{backend: Backend::CPU, id: 0};
    let optimizer = get_optimizer_with_defaults(optimizer_type).unwrap();
    let loss_fct = "mse";
    let mut model = Box::new(Sequential::new(manager.clone()
                                             , optimizer        // optimizer
                                             , loss_fct         // loss
                                             , gpu_device));    // device for model

    // Add the unitary layer
    model.add::<f32>("unitary", hashmap!["input_size"   => 2.to_string()
                     , "output_size"  => 1.to_string()
                     , "hidden_size"  => hidden_dims.to_string()
                     , "o_activation" => "ones".to_string()
                     , "h_init"       => "glorot_uniform".to_string()
                     , "v_init"       => "glorot_uniform".to_string()
                     , "phase_init"      => "glorot_uniform".to_string()
                     , "householder_init"      => "glorot_uniform".to_string()
                     , "u_init"       => "glorot_uniform".to_string()
                     , "h_bias_init"      => "zeros".to_string()
                     , "o_bias_init"      => "zeros".to_string()]);


    model.info();
    manager.swap_device(cpu_device);

    let train_generator = AddingProblemSource::new(batch_size
                                                   , bptt_unroll
                                                   , DType::F32
                                                   , num_train_samples);

    // Training process
    let loss = model.fit::<AddingProblemSource, f32>(&train_generator
                                                     , cpu_device
                                                     , epochs
                                                     , batch_size
                                                     , Some(bptt_unroll)
                                                     , Some(&loss_indices)
                                                     , true);
    println!(" ");

    // Testing process
    manager.swap_device(cpu_device);

    let test_generator = AddingProblemSource::new(num_test_samples
                                                  , bptt_unroll
                                                  , DType::F32
                                                  , num_test_samples);









    let minibatch = test_generator.get_test_iter(num_test_samples);
    let batch_input = manager.swap_array_backend::<f32>(&minibatch.input.into_inner()
                                                        , cpu_device
                                                        , gpu_device);
    let batch_target = manager.swap_array_backend::<f32>(&minibatch.target.into_inner()
                                                         , cpu_device
                                                         , gpu_device);
    let batch_pred = model.forward::<f32>(&batch_input, gpu_device, gpu_device);

    let dims = Dim4::new(&[num_test_samples,1,1,1]);
    let mut wins_vec = af::constant(1f32, dims);

    let pred = batch_pred[(bptt_unroll - 1) as usize].clone();
    let tar = af::slice(&batch_target, bptt_unroll - 1);
    af::print(&pred);
    af::print(&tar);
    
    // Computes loss
    let avg_loss = loss::get_loss(loss_fct, &pred, &tar).unwrap();
}

