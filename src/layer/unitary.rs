use num;
use af;
use af::{Array, MatProp, Dim4};

use std::sync::{Arc, Mutex};
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
        let h_size = h.dims()[1] as i64;
        let h_size_f = h_size as f64;
 

        // D1*h
        let D = af::cplx2(&af::cos(&params.weights[1])
                                    , &af::sin(&params.weights[1])
                                    ,false);
        h = af::mul(&D, &h, true);


        // F*
        h = af::transpose(&af::fft(&af::transpose(&h, false), 1.0, h_size)
                              , false);


        // R1*
        let h_temp = h.clone();
        let mut r = af::Array::new(&[num::pow(af::norm(&params.weights[4], af::NormType::VECTOR_2, 1., 1.), 2) as f32]
                               , Dim4::new(&[1,1,1,1])).clone();
        r = af::div(&params.weights[4], &r, true);
        h = af::matmul(&h
                       , &r
                       , MatProp::NONE
                       , MatProp::NONE);
        h = af::matmul(&h
                       , &r
                       , MatProp::NONE
                       , MatProp::CTRANS);
        h = af::sub(&h_temp, &h, false);


        // Pi*
        h = af::lookup(&h, &params.optional[0], 1);


        // D2*
        let D = af::cplx2(&af::cos(&params.weights[2])
                                    , &af::sin(&params.weights[2])
                                    ,false);
        h = af::mul(&D, &h, true);


        // F-1
        h = af::transpose(&af::ifft(&af::transpose(&h, false), 1.0/h_size_f, h_size)
                              , false);

        // R2
        let h_temp = h.clone();
        let mut r = af::Array::new(&[num::pow(af::norm(&params.weights[5], af::NormType::VECTOR_2, 1., 1.), 2) as f32]
                                   , Dim4::new(&[1,1,1,1]));
        r = af::div(&params.weights[5], &r, true);
        h = af::matmul(&h
                       , &r
                       , MatProp::NONE
                       , MatProp::NONE);
        h = af::matmul(&h
                       , &r
                       , MatProp::NONE
                       , MatProp::CTRANS);
        h = af::sub(&h_temp, &h, false);


        // D3
        let D = af::cplx2(&af::cos(&params.weights[3])
                                    , &af::sin(&params.weights[3])
                                    ,false);
        h = af::mul(&D, &h, true);
        h
    }
}

impl Layer for Unitary
{
    fn forward(&self, params:  Arc<Mutex<Params>>, inputs: &Input, train: bool) -> Input
    {
        let mut ltex = params.lock().unwrap();

        let t = ltex.current_unroll;
      
        // we compute h_t+1 = sigma1(W*h_t + V*x_t + b1) 
        let wh = Unitary::wh(&ltex);

        // In order to convert inputs.data into a complex array
        let zeros = initializations::zeros::<Complex<f32>>(inputs.data.dims()); 

        let vx = af::matmul(&af::add(&inputs.data, &zeros, false)
                             , &ltex.weights[0]
                             , MatProp::NONE
                             , MatProp::NONE);
        af::print(&vx);
        let new_h = af::add(&af::add(&wh, &vx, false)
                            , &ltex.biases[0]
                            , true);
        let sigma_result = activations::get_activation(&ltex.activations[0]
                                                           , &new_h).unwrap();
        ltex.recurrences.push(sigma_result);


        // we compute o_t = sigma2(U*h_t + b2)
        let r_h = af::real(&ltex.recurrences[t]);
        let c_h = af::imag(&ltex.recurrences[t]);
        let concat_h = af::join(1, &r_h, &c_h);
        let uh = af::matmul(&concat_h
                            , &ltex.weights[6]
                            , MatProp::NONE
                            , MatProp::NONE);

        let new_o = af::add(&uh, &ltex.biases[1], true);
        
        let out = Input {
            data: activations::get_activation(&ltex.activations[1]
                                             , &new_o).unwrap()
            , activation: ltex.activations[1].clone()
        };
        ltex.outputs.push(out.clone());
        out


        
    }

    fn backward(&self, params: Arc<Mutex<Params>>, delta: &Array) -> Array {
        
        // Not implemented yet
        let dim = Dim4::new(&[1,1,1,1]);
        af::Array::new(&[0], dim)
    }
}

