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

fn r_d(param: Array, ar: Array) -> Array {
    let D = af::cplx2(&af::cos(&param)
                      , &af::mul(&af::sin(&param), &-1, true)
                      ,false);
    af::mul(&D, &ar, true)
}

fn h_fft(ar: Array) -> Array {
        let ar_size = ar.dims()[1];
        af::transpose(&af::fft(&af::transpose(&ar, false), 1.0, ar_size as i64)
                      , false)
}

fn r_fft(ar: Array) -> Array {
        let ar_size = ar.dims()[1];
        af::transpose(&af::ifft(&af::transpose(&ar, false), 1.0, ar_size as i64)
                      , false)
}

fn h_ifft(ar: Array) -> Array {
        let ar_size = ar.dims()[1];
        af::transpose(&af::ifft(&af::transpose(&ar, false), 1.0/(ar_size as f64), ar_size as i64)
                              , false)
}

fn r_ifft(ar: Array) -> Array {
        let ar_size = ar.dims()[1];
        af::transpose(&af::fft(&af::transpose(&ar, false), 1.0/(ar_size as f64), ar_size as i64)
                              , false)
}

fn h_pi(param: Array, ar: Array) -> Array {
        af::lookup(&ar, &param, 1)
}

fn h_r(param: Array, mut ar: Array) -> Array {
        let ar_temp = ar.clone();
        ar = af::matmul(&param
                       , &ar
                       , MatProp::NONE
                       , MatProp::TRANS);
        ar = af::matmul(&ar
                       , &af::conjg(&param)
                       , MatProp::TRANS
                       , MatProp::NONE);
        ar = af::mul(&ar, &2, true);
        ar = af::sub(&ar_temp, &ar, false);
        ar

}

// Wh = (D3 R2 F-1 D2 Pi R1 F D1) h
fn wh(p1: Array, p2: Array, p3: Array, p4: Array, p5: Array, p6: Array, ar: Array) -> Array { 
    let mut current = ar;
    current = h_d(p1, current);
    current = h_fft(current);
    current = h_r(p2, current);
    current = h_pi(p3, current);
    current = h_d(p4, current);
    current = h_ifft(current);
    current = h_r(p5, current);
    current = h_d(p6, current);
    current
           
}

fn to_complex(ar:Array) -> Array {  
    let dim = ar.dims()[1];
    af::cplx2(&af::cols(&ar, 0, dim/2-1), &af::cols(&ar, dim/2, dim-1), false)
}

fn to_real(ar: Array) -> Array {
    af::join(1, &af::real(&ar), &af::imag(&ar))
}


impl Layer for Unitary
{
    fn forward(&self, params:  Arc<Mutex<Params>>, inputs: &Array, state: Option<Array>) -> (Array, Option<Array>)
    {
        let mut ltex = params.lock().unwrap();
        let t = ltex.current_unroll;
        
        // Transformation of complex parameters
        let mut weight0 = to_complex(ltex.weights[0].clone());
        let mut weight4 = to_complex(ltex.weights[4].clone());
        let mut weight5 = to_complex(ltex.weights[5].clone());
        let mut bias0 = to_complex(ltex.biases[0].clone());
        let mut rec_t0 = to_complex(ltex.recurrences[t].clone());

        // Not complex
        let mut weight1 = ltex.weights[1].clone();
        let mut weight2 = ltex.weights[2].clone();
        let mut weight3 = ltex.weights[3].clone();
        let mut weight6 = ltex.weights[6].clone();
        let mut bias1 = ltex.biases[1].clone();

        if t == 0 {
            // Make a copy of h0 for all batch inputs
            let hidden_size = ltex.weights[0].dims()[1];
            let batch_size = inputs.dims()[0];
            let zero = af::constant(0f32, Dim4::new(&[batch_size, hidden_size, 1, 1]));
            ltex.recurrences[0] = af::add(&zero, &ltex.recurrences[0], true);
            rec_t0 = to_complex(ltex.recurrences[0].clone());

            // We normalize Householder parameters;
            let sqrNorm = af::norm(&weight4, af::NormType::VECTOR_2, 1., 1.)as f32;
            weight4 = af::div(&weight4, &sqrNorm, true);
            ltex.weights[4] = to_real(weight4.clone());
            
            let sqrNorm = af::norm(&weight5, af::NormType::VECTOR_2, 1., 1.)as f32;
            weight5 = af::div(&weight5, &sqrNorm, true);
            ltex.weights[5] = to_real(weight5.clone());
        }
        
        
        
      
        // we compute h_t+1 = sigma1(W*h_t + V*x_t + b1) 
        let wh = wh(weight1.clone()
                    , weight4.clone()
                    , ltex.optional[0].clone()
                    , weight2.clone()
                    , weight5.clone()
                    , weight3.clone()
                    , rec_t0.clone());

        // In order to convert inputs.data into a complex array
        let c_zeros = initializations::zeros::<Complex<f32>>(inputs.dims()); 

        let c_inputs = af::add(inputs, &c_zeros, false);
        let vx = af::matmul(&c_inputs
                             , &weight0
                             , MatProp::NONE
                             , MatProp::NONE);

        let new_h = af::add(&af::add(&wh, &vx, true)
                            , &bias0
                            , true);

        let sigma_result = activations::get_activation(&ltex.activations[0]
                                                       , &new_h).unwrap();


        // we compute o_t = sigma2(U*h_t + b2)
        let r_h = af::real(&sigma_result);
        let c_h = af::imag(&sigma_result);
        let concat_h = af::join(1, &r_h, &c_h);
        let uh = af::matmul(&concat_h
                            , &weight6
                            , MatProp::NONE
                            , MatProp::NONE);

        let new_o = af::add(&uh, &bias1, true);
       
         
        let out = activations::get_activation(&ltex.activations[1]
                                              , &new_o).unwrap(); 
        

        if ltex.inputs.len() > t {
            ltex.inputs[t] = c_inputs.clone();
            ltex.recurrences[t+1] = to_real(sigma_result.clone());
            ltex.outputs[t] = out.clone();
        }
        else{
            ltex.inputs.push(c_inputs.clone());
            ltex.recurrences.push(to_real(sigma_result.clone()));
            ltex.outputs.push(out.clone());
        }
        //println!("{}", &(af::norm(&ltex.recurrences[t], af::NormType::VECTOR_2, 1.,1.)as f32));
        ltex.current_unroll += 1;

        (out.clone(), None)
    }



    fn backward(&self, params: Arc<Mutex<Params>>, delta: &Array) -> Array {
        let mut ltex = params.lock().unwrap();
        ltex.current_unroll -= 1;
        let t = ltex.current_unroll;
        let t_max = ltex.outputs.len()-1;
        
        // Transformation of complex parameters
        let mut weight0 = to_complex(ltex.weights[0].clone());
        let mut weight4 = to_complex(ltex.weights[4].clone());
        let mut weight5 = to_complex(ltex.weights[5].clone());
        let mut bias0 = to_complex(ltex.biases[0].clone());
        let mut rec_t0 = to_complex(ltex.recurrences[t].clone());
        let mut rec_t1 = to_complex(ltex.recurrences[t+1].clone());

        let mut delta0 = to_complex(ltex.deltas[0].clone());
        let mut delta4 = to_complex(ltex.deltas[4].clone());
        let mut delta5 = to_complex(ltex.deltas[5].clone());
        let mut delta7 = to_complex(ltex.deltas[7].clone());

        // Not complex
        let mut weight1 = ltex.weights[1].clone();
        let mut weight2 = ltex.weights[2].clone();
        let mut weight3 = ltex.weights[3].clone();
        let mut weight6 = ltex.weights[6].clone();
        let mut bias1 = ltex.biases[1].clone();

        let mut delta1 = ltex.deltas[1].clone();
        let mut delta2 = ltex.deltas[2].clone();
        let mut delta3 = ltex.deltas[3].clone();
        let mut delta6 = ltex.deltas[6].clone();
        let mut delta8 = ltex.deltas[8].clone();

        if t == t_max {
            // We normalize Householder parameters;
            let sqrNorm = af::norm(&weight4, af::NormType::VECTOR_2, 1., 1.)as f32;
            weight4 = af::div(&weight4, &sqrNorm, true);
            ltex.weights[4] = to_real(weight4.clone());
            
            let sqrNorm = af::norm(&weight5, af::NormType::VECTOR_2, 1., 1.)as f32;
            weight5 = af::div(&weight5, &sqrNorm, true);
            ltex.weights[5] = to_real(weight5.clone());
        }

        let p1 = weight1.clone();
        let p2 = weight4.clone();
        let p3 = ltex.optional[0].clone();
        let p4 = weight2.clone();
        let p5 = weight5.clone();
        let p6 = weight3.clone();

        let dim_h = rec_t0.dims()[1];
        assert!(t >= 0
                , "Cannot call backward pass without at least 1 forward pass");
       

        // We write d_ to say dL/d_
        // do => dz2
        
        let d_z2 = af::mul(delta
                         , &activations::get_derivative(&ltex.activations[1], &ltex.outputs[t]).unwrap()
                          , false);
        
        // dz2 => dh_{t}
        let prod = af::matmul(&d_z2, &weight6, MatProp::NONE, MatProp::TRANS);
        let d_h1 = af::cplx2(&af::cols(&prod, 0, dim_h-1)
                            , &af::cols(&prod, dim_h, 2*dim_h-1)
                            , false);

        let mut d_rec = d_h1.clone();
        if t == t_max {
            // Check to see if we already have a state derivative, else add one
            if ltex.state_derivatives.len() == 0 {
                ltex.state_derivatives.push(af::constant(0, Dim4::new(&[1, 1, 1, 1])));
            }
            ltex.state_derivatives[0]= to_real(d_rec.clone());
        }
        else {
            // dh_{t+1} => dh_{t}
            d_rec = to_complex(ltex.state_derivatives[0].clone());
            let mut rec_t2 = to_complex(ltex.recurrences[t+2].clone());
            let d_activ = activations::get_derivative(&ltex.activations[0], &rec_t2).unwrap();
            let d_h2 = af::mul(&r_d(p1.clone()
                                    , r_fft(h_r(p2.clone()
                                                , h_pi(p3.clone()
                                                       , r_d(p4.clone()
                                                             , r_ifft(h_r(p5.clone()
                                                                          , r_d(p6.clone(), d_rec.clone()))))))))
                               , &d_activ
                               , false);
            // dz2 & dh_{t+1} => dh_{t}
            d_rec = af::add(&d_h1, &d_h2, false);
            ltex.state_derivatives[0] = to_real(d_rec.clone());
        }

        // dh_{t} => dz
        let d_activ = activations::get_derivative(&ltex.activations[0], &rec_t1).unwrap();
        let d_z = af::mul(&d_rec, &d_activ, false);

         
        
        //-----------------------------------------------------------------------------
        // dz => dW
        // dD

        // D1
        let dd1_left = rec_t0.clone();
        let dd1_right = r_fft(h_r(p2.clone()
                                  , h_pi(p3.clone()
                                         , r_d(p4.clone()
                                               , r_ifft(h_r(p5.clone()
                                                            , r_d(p6.clone(), d_z.clone())))))));
        // We add the derivatives from the real and imaginary parts
        let dd1_sin = af::sin(&weight1);
        let dd1_cos = af::cos(&weight1);
        let dd1_real = af::mul(&af::add(&af::mul(&af::real(&dd1_left), &dd1_sin, true), &af::mul(&af::imag(&dd1_left), &dd1_cos, true), false), &-1, true);
        let dd1_imag = af::sub(&af::mul(&af::real(&dd1_left), &dd1_cos, true), &af::mul(&af::imag(&dd1_left), &dd1_sin, true), false);
        let dd1_phase = af::add(&af::mul(&dd1_real, &af::real(&dd1_right), false)
                                , &af::mul(&dd1_imag, &af::imag(&dd1_right), false)
                                , false);
        ltex.deltas[1] = af::add(&delta1, &af::sum(&dd1_phase, 0), false);
        
        // D2
        let dd2_left = h_pi(p3.clone()
                            , h_r(p2.clone()
                                  , h_fft(h_d(p1.clone(), rec_t0.clone()))));   
        let dd2_right = r_ifft(h_r(p5.clone()
                                   , r_d(p6.clone(), d_z.clone())));
        let dd2_sin = af::sin(&weight2);
        let dd2_cos = af::cos(&weight2);
        let dd2_real = af::mul(&af::add(&af::mul(&af::real(&dd2_left), &dd2_sin, true), &af::mul(&af::imag(&dd2_left), &dd2_cos, true), false), &-1, true);
        let dd2_imag = af::sub(&af::mul(&af::real(&dd2_left), &dd2_cos, true), &af::mul(&af::imag(&dd2_left), &dd2_sin, true), false);
        let dd2_phase = af::add(&af::mul(&dd2_real, &af::real(&dd2_right), false)
                                , &af::mul(&dd2_imag, &af::imag(&dd2_right), false)
                                , false);
        ltex.deltas[2] = af::add(&delta2, &af::sum(&dd2_phase, 0), false);

        // D3
        let dd3_left = h_r(p5.clone()
                           , h_ifft(h_d(p4.clone()
                                        , h_pi(p3.clone()
                                               , h_r(p2.clone()
                                                     , h_fft(h_d(p1.clone(), rec_t0.clone())))))));
        let dd3_right = d_z.clone();
        let dd3_sin = af::sin(&weight3);
        let dd3_cos = af::cos(&weight3);
        let dd3_real = af::mul(&af::add(&af::mul(&af::real(&dd3_left), &dd3_sin, true), &af::mul(&af::imag(&dd3_left), &dd3_cos, true), false), &-1, true);
        let dd3_imag = af::sub(&af::mul(&af::real(&dd3_left), &dd3_cos, true), &af::mul(&af::imag(&dd3_left), &dd3_sin, true), false);
        let dd3_phase = af::add(&af::mul(&dd3_real, &af::real(&dd3_right), false)
                                , &af::mul(&dd3_imag, &af::imag(&dd3_right), false)
                                , false);
        ltex.deltas[3] = af::add(&delta3, &af::sum(&dd3_phase, 0), false);

        
        //------------------------------------------------------------------------------
        // dR
        
        // R1
        let dr1_left = h_fft(h_d(p1.clone(), rec_t0.clone()));
        let dr1_right = h_pi(p3.clone()
                         , r_d(p4.clone()
                         , r_ifft(h_r(p5.clone()
                                      , r_d(p6.clone(), d_z.clone())))));
        
        let w = weight4.clone();
        let dh = dr1_right.clone();
        let dh2 = dh.clone();
        let h0 = dr1_left.clone();
        let h1 = af::matmul(&w, &h0, MatProp::NONE, MatProp::TRANS);
        let h2 = af::matmul(&h1, &af::conjg(&w), MatProp::TRANS, MatProp::NONE);

        let dh1 = af::transpose(&af::matmul(&w, &dh2, MatProp::NONE, MatProp::TRANS), false);
        let dr11 = af::mul(&af::conjg(&h0), &dh1, true);
        
        let dr12 = af::conjg(&af::mul(&af::transpose(&af::conjg(&h1), false), &dh2, true));

        let dh3 = af::sum(&af::mul(&dh, &af::conjg(&h2), false), 1);
        let dr13 = af::matmul(&dh3 , &w, MatProp::NONE, MatProp::NONE);
        
        let dr14 = af::conjg(&af::matmul(&dh3 , &af::conjg(&w), MatProp::NONE, MatProp::NONE));

        let dr1 = af::mul(&af::sub(&af::sub(&af::add(&dr11, &dr12, false), &dr13, false), &dr14, false), &-2, true);

        delta4 = af::add(&delta4, &af::sum(&dr1, 0), false);
        ltex.deltas[4] = to_real(delta4.clone());

        // R2

        let dr2_left = h_ifft(h_d(p4.clone()
                                  , h_pi(p3.clone()
                                         , h_r(p2.clone()
                                               , h_fft(h_d(p1.clone(), rec_t0.clone()))))));
        let dr2_right = r_d(p6.clone(), d_z.clone());

        let w = weight5.clone();
        let dh = dr2_right.clone();
        let dh2 = dh.clone();
        let h0 = dr2_left.clone();
        let h1 = af::matmul(&w, &h0, MatProp::NONE, MatProp::TRANS);
        let h2 = af::matmul(&h1, &af::conjg(&w), MatProp::TRANS, MatProp::NONE);

        let dh1 = af::transpose(&af::matmul(&w, &dh2, MatProp::NONE, MatProp::TRANS), false);
        let dr21 = af::mul(&af::conjg(&h0), &dh1, true);
        
        let dr22 = af::conjg(&af::mul(&af::transpose(&af::conjg(&h1), false), &dh2, true));

        let dh3 = af::sum(&af::mul(&dh, &af::conjg(&h2), false), 1);
        let dr23 = af::matmul(&dh3 , &w, MatProp::NONE, MatProp::NONE);
        
        let dr24 = af::conjg(&af::matmul(&dh3 , &af::conjg(&w), MatProp::NONE, MatProp::NONE));

        let dr2 = af::mul(&af::sub(&af::sub(&af::add(&dr21, &dr22, false), &dr23, false), &dr24, false), &-2, true);

        delta5 = af::add(&delta5, &af::sum(&dr2, 0), false);
        ltex.deltas[5] = to_real(delta5.clone());


       
        
        // TO DO : fix the name of parameters to be coherent with the one of params.rs 
        //-----------------------------------------------------------------------------
        // dz => dU
        let d_u = af::matmul(&ltex.inputs[t]
                         , &d_z
                         , MatProp::CTRANS
                         , MatProp::NONE);
        delta0 = af::add(&delta0, &d_u, false);
        ltex.deltas[0] = to_real(delta0.clone());

        
        //-----------------------------------------------------------------------------
        // dz => db
        delta7 = af::add(&delta7, &af::sum(&d_z, 0), false);
        ltex.deltas[7] = to_real(delta7.clone());


        
        //-----------------------------------------------------------------------------
        // dz2 => db2
        ltex.deltas[8] = af::add(&delta8, &af::sum(&d_z2, 0), false);

        
        //-----------------------------------------------------------------------------
        // dz2 => dV
        let concat_h = af::join(1,
                                &af::real(&rec_t1)
                                , &af::imag(&rec_t1));
        let d_v = af::matmul(&concat_h
                             , &d_z2
                             , MatProp::TRANS
                             , MatProp::NONE);
        ltex.deltas[6] = af::add(&delta6, &d_v, false);
        

        
        //-----------------------------------------------------------------------------
        // dh_{t} => dx
        af::real(&af::matmul(&d_rec, &weight0, MatProp::NONE, MatProp::TRANS))
    }
}

