#[macro_use] extern crate hal;
extern crate arrayfire as af;
extern crate rand;
use rand::distributions::{Range, IndependentSample};

use hal::utils;
use hal::Model;
use hal::initializations::uniform;
use hal::optimizer::{Optimizer, get_optimizer_with_defaults};
use hal::data::{DataSource, AddingProblemSource, CopyingProblemSource};
use hal::error::HALError;
use hal::model::{Sequential};
use hal::plot::{plot_vec, plot_array};
use hal::device::{DeviceManagerFactory, Device};
use af::{Dim4, Backend, HasAfEnum, MatProp, DType};


fn main() {
    let dimA = Dim4::new(&[2,2,1,1]);

    let rA = af::Array::new(&[1, 1, 1, 1], dimA);
    let iA = af::Array::new(&[2, 2, 2, 2], dimA);
    let mut a = af::cplx2(&rA, &iA, false);
    let rB = af::Array::new(&[2, 2, 2, 2], dimA);
    let iB = af::Array::new(&[2, 2, 2, 2], dimA);
    let mut b = af::cplx2(&rB, &iB, false);

    af::print(&a);

    let b = 2f32;
    a = af::div (&b, &a, false);

    println!("{}", &a.dims(2));




}




