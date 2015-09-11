extern crate arrayfire as af;
extern crate nalgebra as na;
extern crate rand;

pub use layer::{Layer};
pub mod layer;

pub use model::{Model};
pub mod model;

pub use optimizer::{Optimizer};
pub mod optimizer;

pub mod error;
pub mod loss;
pub mod activations;
pub mod initializations;
pub mod plot;
pub mod utils;
