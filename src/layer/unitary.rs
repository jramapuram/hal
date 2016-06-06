use num;
use af;
use af::{Array, MatProp, Dim4};

use activations;
use initializations;
use params::{Input, Params};
use layer::Layer;

use num::Complex;


pub struct Unitary {
    pub input_size: usize,
    pub output_size: usize,
}
impl Unitary
{
    // Wh = (D3 R2 F-1 D2 Pi R1 F D1) h
    fn wh(params: &Params) -> Array {
        let t = params.current_unroll;
        let mut h = params.recurrences[t].clone();
        let h_size = h.dims().unwrap()[1] as i64;
        let h_size_f = h_size as f64;
 

        // D1*h
        let D = af::cplx2(&af::cos(&params.weights[1]).unwrap()
                                    , &af::sin(&params.weights[1]).unwrap()
                                    ,false).unwrap();
        h = af::mul(&D, &h, true).unwrap();


        // F*
        h = af::transpose(&af::fft(&af::transpose(&h, false).unwrap(), 1.0, h_size).unwrap()
                              , false).unwrap();


        // R1*
        let h_temp = h.clone();
        let r = af::Array::new(&[af::norm(&params.weights[4], af::NormType::VECTOR_2, 1., 1.).unwrap() as f32]
                               , Dim4::new(&[1,1,1,1])).unwrap().clone();
        let r = af::div(&params.weights[4], &r, true).unwrap();
        h = af::matmul(&h
                       , &r
                       , MatProp::NONE
                       , MatProp::NONE).unwrap();
        h = af::matmul(&h
                       , &r
                       , MatProp::NONE
                       , MatProp::CTRANS).unwrap();
        h = af::sub(&h_temp, &h, false).unwrap();


        // Pi*
        h = af::lookup(&h, &params.optional[0], 1).unwrap();


        // D2*
        let D = af::cplx2(&af::cos(&params.weights[2]).unwrap()
                                    , &af::sin(&params.weights[2]).unwrap()
                                    ,false).unwrap();
        h = af::mul(&D, &h, true).unwrap();


        // F-1
        h = af::transpose(&af::ifft(&af::transpose(&h, false).unwrap(), 1.0/h_size_f, h_size).unwrap()
                              , false).unwrap();

        // R2
        let h_temp = h.clone();
        let mut r = af::Array::new(&[af::norm(&params.weights[5], af::NormType::VECTOR_2, 1., 1.).unwrap() as f32]
                                   , Dim4::new(&[1,1,1,1])).unwrap();
        r = af::div(&params.weights[5], &r, true).unwrap();
        h = af::matmul(&h
                       , &r
                       , MatProp::NONE
                       , MatProp::NONE).unwrap();
        h = af::matmul(&h
                       , &r
                       , MatProp::NONE
                       , MatProp::CTRANS).unwrap();
        h = af::sub(&h_temp, &h, false).unwrap();


        // D3
        let D = af::cplx2(&af::cos(&params.weights[3]).unwrap()
                                    , &af::sin(&params.weights[3]).unwrap()
                                    ,false).unwrap();
        h = af::mul(&D, &h, true).unwrap();
        h
    }
}

impl Layer for Unitary
{
    fn forward(&self, params: &mut Params, inputs: &Input, train: bool) -> Input
    {
        let t = params.current_unroll;
      
        // we compute h_t+1 = sigma1(W*h_t + V*x_t + b1) 
        let wh = Unitary::wh(&params);

        // In order to convert inputs.data into a complex array
        let zeros = initializations::zeros::<Complex<f32>>(inputs.data.dims().unwrap()); 

        let vx = af::matmul(&af::add(&inputs.data, &zeros, false).unwrap()
                             , &params.weights[0]
                             , MatProp::NONE
                             , MatProp::NONE).unwrap();
        af::print(&vx);
        let new_h = af::add(&af::add(&wh, &vx, false).unwrap()
                            , &params.biases[0]
                            , true).unwrap();
        params.recurrences.push(activations::get_activation(&params.activations[0]
                                                           , &new_h).unwrap());


        // we compute o_t = sigma2(U*h_t + b2)
        let r_h = af::real(&params.recurrences[t]).unwrap();
        let c_h = af::imag(&params.recurrences[t]).unwrap();
        let concat_h = af::join(1, &r_h, &c_h).unwrap();
        let uh = af::matmul(&concat_h
                            , &params.weights[6]
                            , MatProp::NONE
                            , MatProp::NONE).unwrap();

        let new_o = af::add(&uh, &params.biases[1], true).unwrap();
        
        let out = Input {
            data: activations::get_activation(&params.activations[1]
                                             , &new_o).unwrap()
            , activation: params.activations[1].clone()
        };
        params.outputs.push(out.clone());
        out


        
    }

    fn backward(&self, params: &mut Params, delta: &Array) -> Array {
        let dim = Dim4::new(&[1,1,1,1]);
        af::Array::new(&[0], dim).unwrap()
    }
}

