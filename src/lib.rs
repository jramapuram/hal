extern crate arrayfire as af;

pub use layer::{Layer, Params};
pub mod layer;

pub use model::{Model};
pub mod model;

pub use optimizer::{Optimizer};
pub mod optimizer;
