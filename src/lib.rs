extern crate arrayfire as af;

pub use layer::{Layer, ArrayVector};
pub mod layer;

pub use model::{Model};
pub mod model;

pub use optimizer::{Optimizer};
pub mod optimizer;

pub mod error;
pub mod loss;
pub mod activations;
pub mod initializations;
