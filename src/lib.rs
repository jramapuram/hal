extern crate arrayfire as af;

pub use layer::{Layer, Params};
mod layer;

pub use model::{Model, Sequential};
mod model;
//mod optimizer;
