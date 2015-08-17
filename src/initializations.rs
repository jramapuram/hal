use af::Dim4;
use af::Array;

pub fn normal(dims: &Dim4) -> Array {
  af::randn(dims, af::Aftype::F32).unwrap()
}

pub fn uniform(dims: &Dim4, spread: f32){
  af::randu(dim, af::Aftype::F32).and_then(|x| x * spread - spread / 2).unwrap()
}

pub fn zeros(dims: &Dim4) -> Array {
  af::constant(0.0 as f32, dims).unwrap()
}

pub fn ones(dims: &Dim4) -> Array {
  af::constant(1.0 as f32, dims).unwrap()
}

pub fn get_initialization(name: &str, dims: &Dim4) -> Array {
  match(name){
    "normal"  => normal(dims),
    "uniform" => uniform(dims, 0.05),
    "zeros"   => zeros(dims),
    "ones"    => ones(dims),
    _         => HALError::UNKNOWN,
  }
}
