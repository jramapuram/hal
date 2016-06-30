use num;
use af;
use af::{Array, MatProp, Dim4, DType};

use std::sync::{Arc, Mutex};
use activations;
use initializations;
use params::{Params};
use layer::Layer;

use num::Complex;


pub struct Unitary {
    pub input_size: usize,
    pub output_size: usize,
}

fn h_d(param: Array, ar: Array) -> Array {
    let D = af::cplx2(&af::cos(&param)
                      , &af::sin(&param)
                      ,false);
    af::mul(&D, &ar, true)
}

fn h_fft(ar: Array) -> Array {
        let ar_size = ar.dims()[1];
        af::transpose(&af::fft(&af::transpose(&ar, false), 1.0, ar_size as i64)
                      , false)
}

fn h_ifft(ar: Array) -> Array {
        let ar_size = ar.dims()[1];
        af::transpose(&af::ifft(&af::transpose(&ar, false), 1.0/(ar_size as f64), ar_size as i64)
                              , false)
}

fn h_pi(param: Array, ar: Array) -> Array {
        af::lookup(&ar, &param, 1)
}

fn h_r(param: Array, mut ar: Array) -> Array {
        let ar_temp = ar.clone();
        ar = af::matmul(&ar
                       , &param
                       , MatProp::NONE
                       , MatProp::CTRANS);
        ar = af::matmul(&ar
                       , &param
                       , MatProp::NONE
                       , MatProp::NONE);
        af::sub(&ar_temp, &ar, false)

}

// Wh = (D3 R2 F-1 D2 Pi R1 F D1) h
fn wh(p1: Array, p2: Array, p3: Array, p4: Array, p5: Array, p6: Array, ar: Array) -> Array { 
        h_d(p6
            , h_r(p5
                  , h_ifft(h_d(p4
                               , h_pi(p3
                                    , h_r(p2
                                          , h_fft(h_d(p1, ar))))))))
           
}

impl Layer for Unitary
{
    fn forward(&self, params:  Arc<Mutex<Params>>, inputs: &Array) -> Array
    {
        let mut ltex = params.lock().unwrap();
        let t = ltex.current_unroll;

        if t == 0 {
            // Make a copy of h0 for all batch inputs
            let hidden_size = ltex.weights[0].dims()[1];
            let batch_size = inputs.dims()[0];
            let zero = af::constant(0f32, Dim4::new(&[batch_size, hidden_size, 1, 1]));
            ltex.recurrences[0] = af::add(&zero, &ltex.recurrences[0], true);

            // We normalize Householder parameters;
            let sqrNorm = af::norm(&ltex.weights[4], af::NormType::VECTOR_2, 1., 1.)as f32;
            ltex.weights[4] = af::div(&ltex.weights[4], &sqrNorm, true);
            
            let sqrNorm = af::norm(&ltex.weights[5], af::NormType::VECTOR_2, 1., 1.)as f32;
            ltex.weights[5] = af::div(&ltex.weights[5], &sqrNorm, true);
        }

      
        // we compute h_t+1 = sigma1(W*h_t + V*x_t + b1) 
        let wh = wh(ltex.weights[1].clone()
                    , ltex.weights[4].clone()
                    , ltex.optional[0].clone()
                    , ltex.weights[2].clone()
                    , ltex.weights[5].clone()
                    , ltex.weights[3].clone()
                    , ltex.recurrences[t].clone());

        // In order to convert inputs.data into a complex array
        let c_zeros = initializations::zeros::<Complex<f32>>(inputs.dims()); 

        let c_inputs = af::add(inputs, &c_zeros, false);
        let vx = af::matmul(&c_inputs
                             , &ltex.weights[0]
                             , MatProp::NONE
                             , MatProp::NONE);

        let new_h = af::add(&af::add(&wh, &vx, true)
                            , &ltex.biases[0]
                            , true);
        let sigma_result = activations::get_activation(&ltex.activations[0]
                                                       , &new_h).unwrap();

        if ltex.inputs.len() > t {
            ltex.recurrences[t+1] = sigma_result.clone();
        }
        else {
            ltex.recurrences.push(sigma_result.clone());
        }

        // we compute o_t = sigma2(U*h_t + b2)
        let r_h = af::real(&ltex.recurrences[t+1]);
        let c_h = af::imag(&ltex.recurrences[t+1]);
        let concat_h = af::join(1, &r_h, &c_h);
        let uh = af::matmul(&concat_h
                            , &ltex.weights[6]
                            , MatProp::NONE
                            , MatProp::NONE);

        let new_o = af::add(&uh, &ltex.biases[1], true);
        
        let out = activations::get_activation(&ltex.activations[1]
                                              , &new_o).unwrap(); 
        if ltex.inputs.len() > t {
            ltex.inputs[t] = c_inputs.clone();
            ltex.outputs[t] = out.clone();
        }
        else{
            ltex.inputs.push(c_inputs.clone());
            ltex.outputs.push(out.clone());
        }
        ltex.current_unroll += 1;
        out.clone()
    }



    fn backward(&self, params: Arc<Mutex<Params>>, delta: &Array) -> Array {
        let mut ltex = params.lock().unwrap();
        ltex.current_unroll -= 1;

        let p1 = ltex.weights[1].clone();
        let p2 = ltex.weights[4].clone();
        let p3 = ltex.optional[0].clone();
        let p4 = ltex.weights[2].clone();
        let p5 = ltex.weights[5].clone();
        let p6 = ltex.weights[3].clone();

        let t = ltex.current_unroll;
        let t_max = ltex.outputs.len()-1;
        let dim_h = ltex.recurrences[t].dims()[1];
        assert!(t >= 0
                , "Cannot call backward pass without at least 1 forward pass");
        
        // We write d_ to say dL/d_
        // do => dz2
        let d_z2 = af::mul(delta
                         , &activations::get_derivative(&ltex.activations[1], &ltex.outputs[t]).unwrap()
                          , false);

        // dz2 => dh_{t}
        let prod = af::matmul(&d_z2, &ltex.weights[6], MatProp::NONE, MatProp::TRANS);
        let d_h1 = af::cplx2(&af::cols(&prod, 0, dim_h-1)
                            , &af::cols(&prod, dim_h, 2*dim_h-1)
                            , false);

        if t == t_max {
            ltex.d_rec = d_h1;
        }
        else {
            // dh_{t+1} => dh_{t}
            let d_activ = activations::get_derivative(&ltex.activations[0], &ltex.recurrences[t+2]).unwrap();
            let d_h2 = af::mul(&h_d(p1.clone()
                                    , h_fft(h_r(p2.clone()
                                                , h_pi(p3.clone()
                                                       , h_d(p4.clone()
                                                             , h_ifft(h_r(p5.clone()
                                                                          , h_d(p6.clone(), ltex.d_rec.clone()))))))))
                               , &d_activ
                               , false);
            // dz2 & dh_{t+1} => dh_{t}
            ltex.d_rec = af::add(&d_h1, &d_h2, false);
        }

        // dh_{t} => dz
        let d_activ = activations::get_derivative(&ltex.activations[0], &ltex.recurrences[t+1]).unwrap();
        let d_z = af::mul(&ltex.d_rec, &d_activ, false);

         
        
        //-----------------------------------------------------------------------------
        // dz => dW
        // dD

        // D1
        let dd1_left = ltex.recurrences[t].clone();
        let dd1_right = h_fft(h_r(p2.clone()
                                  , h_pi(p3.clone()
                                         , h_d(p4.clone()
                                               , h_ifft(h_r(p5.clone()
                                                            , h_d(p6.clone(), d_z.clone())))))));
        let dd1 = af::mul(&dd1_left, &dd1_right, false);
        // We add the derivatives from the real and imaginary parts
        let dd1_real = af::mul(&af::real(&dd1), &af::mul(&af::sin(&ltex.weights[1]), &-1, true), true);
        let dd1_imag = af::mul(&af::imag(&dd1), &af::cos(&ltex.weights[1]), true);
        let dd1_phase = af::add(&dd1_real, &dd1_imag, false);
        ltex.deltas[1] = af::add(&ltex.deltas[1], &af::sum(&dd1_phase, 0), false);
        
        // D2
        let dd2_left = h_pi(p3.clone()
                            , h_r(p2.clone()
                                  , h_fft(h_d(p1.clone(), ltex.recurrences[t].clone()))));   
        let dd2_right = h_ifft(h_r(p5.clone()
                                   , h_d(p6.clone(), d_z.clone())));
        let dd2 = af::mul(&dd2_left, &dd2_right, false);
        let dd2_real = af::mul(&af::real(&dd2), &af::mul(&af::sin(&ltex.weights[2]), &-1, true), true);
        let dd2_imag = af::mul(&af::imag(&dd2), &af::cos(&ltex.weights[2]), true);
        let dd2_phase = af::add(&dd2_real, &dd2_imag, false);
        ltex.deltas[2] = af::add(&ltex.deltas[2], &af::sum(&dd2_phase, 0), false);

        // D3
        let dd3_left = h_r(p5.clone()
                           , h_ifft(h_d(p4.clone()
                                        , h_pi(p3.clone()
                                               , h_r(p2.clone()
                                                     , h_fft(h_d(p1.clone(), ltex.recurrences[t].clone())))))));
        let dd3_right = d_z.clone();
        let dd3 = af::mul(&dd3_left, &dd3_right, false);
        let dd3_real = af::mul(&af::real(&dd3), &af::mul(&af::sin(&ltex.weights[3]), &-1, true), true);
        let dd3_imag = af::mul(&af::imag(&dd3), &af::cos(&ltex.weights[3]), true);
        let dd3_phase = af::add(&dd3_real, &dd3_imag, false);
        ltex.deltas[3] = af::add(&ltex.deltas[3], &af::sum(&dd3_phase, 0), false);

        
        //------------------------------------------------------------------------------
        // dR
        
        // R1
        let dr1_left = h_fft(h_d(p1.clone(), ltex.recurrences[t].clone()));
        let dr1_right = h_pi(p3.clone()
                         , h_d(p4.clone()
                         , h_ifft(h_r(p5.clone()
                                      , h_d(p6.clone(), d_z.clone())))));
        let dr1 = af::mul(&af::mul(&af::matmul(&dr1_left
                                               , &ltex.weights[4]
                                               , MatProp::NONE
                                               , MatProp::CTRANS), &-2, true), &dr1_right, true);
        ltex.deltas[4] = af::add(&ltex.deltas[4], &af::sum(&dr1, 0), false);

        // R2
        let dr2_left = h_ifft(h_d(p4.clone()
                                  , h_pi(p3.clone()
                                         , h_r(p2.clone()
                                               , h_d(p1.clone(), ltex.recurrences[t].clone())))));
        let dr2_right = h_d(p6.clone(), d_z.clone());
        let dr2 = af::mul(&af::mul(&af::matmul(&dr2_left
                                               , &ltex.weights[4]
                                               , MatProp::NONE
                                               , MatProp::CTRANS), &-2, true), &dr2_right, true);
        ltex.deltas[5] = af::add(&ltex.deltas[5], &af::sum(&dr2, 0), false);

        
        
        //-----------------------------------------------------------------------------
        // dz => dU
        let d_u = af::matmul(&ltex.inputs[t]
                         , &d_z
                         , MatProp::TRANS
                         , MatProp::NONE);
        ltex.deltas[0] = af::add(&ltex.deltas[0], &d_u, false);

        
        //-----------------------------------------------------------------------------
        // dz => db
        ltex.deltas[7] = af::add(&ltex.deltas[7], &af::sum(&d_z, 0), false);


        
        //-----------------------------------------------------------------------------
        // dz2 => db2
        ltex.deltas[8] = af::add(&ltex.deltas[8], &af::sum(&d_z2, 0), false);

        
        //-----------------------------------------------------------------------------
        // dz2 => dV
        let concat_h = af::join(1,
                                &af::real(&ltex.recurrences[t+1])
                                , &af::imag(&ltex.recurrences[t+1]));
        let d_v = af::matmul(&concat_h
                             , &d_z2
                             , MatProp::TRANS
                             , MatProp::NONE);
        ltex.deltas[6] = af::add(&ltex.deltas[6], &d_v, false);
        

        
        //-----------------------------------------------------------------------------
        // dh_{t} => dx
        af::matmul(&ltex.d_rec, &ltex.weights[0], MatProp::NONE, MatProp::TRANS)
    }
}

