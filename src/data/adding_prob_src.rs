extern crate rand;
use rand::distributions::{Range, IndependentSample};
use af;
use af::{Array, Dim4, MatProp, DType};
use std::cell::{RefCell, Cell};

use initializations::uniform_pos;

use data::{Data, DataSource, DataParams, Normalize, Shuffle};

pub struct AddingProblemSource {
    pub params: DataParams,
    pub iter: Cell<u64>,
    pub offset: Cell<f32>,
    pub bptt_unroll : u64, 
}

impl AddingProblemSource {
    pub fn new(batch_size: u64, bptt_unroll: u64, max_samples: u64, dtype: DType) -> AddingProblemSource
    {
        let input_dims = Dim4::new(&[batch_size, 1, bptt_unroll, 1]);
        let target_dims = Dim4::new(&[batch_size, 1, 1, 1]);
        let train_samples = 0.7 * max_samples as f32;
        let test_samples = 0.2 * max_samples as f32;
        let validation_samples = 0.1 * max_samples as f32;
        AddingProblemSource {
            params : DataParams {
                input_dims: input_dims,
                target_dims: target_dims,
                dtype: dtype,
                normalize: false,
                shuffle: false,
                current_epoch: Cell::new(0),
                num_samples: max_samples,
                num_train: train_samples as u64,
                num_test: test_samples as u64,
                num_validation: Some(validation_samples as u64),
            },
            iter: Cell::new(0),
            offset: Cell::new(0.0f32),
            bptt_unroll: bptt_unroll,
        }
    }

    fn generate_input(&self, batch_size: u64, bptt_unroll: u64) -> Array {
        let dim = Dim4::new(&[batch_size, 1, bptt_unroll/2, 1]);
        let ar1 = uniform_pos::<f32>(dim, 1.0);

        let between1 = Range::new(0, bptt_unroll/4-1);
        let between2 = Range::new(bptt_unroll/4, bptt_unroll/2-1);
        let mut rng1 = rand::thread_rng();
        let mut rng2 = rand::thread_rng();
        let mut ar2 = af::constant(0.0 as f32, dim);
        let one = af::constant(1, Dim4::new(&[1,1,1,1]));

        for i in 0..batch_size {
            let index1 = between1.ind_sample(&mut rng1);
            let index2 = between2.ind_sample(&mut rng2);
            let seqs1 = &[af::Seq::new(i as f64, i as f64, 1.0)
                , af::Seq::new(0.0, 0.0, 1.0)
                , af::Seq::new(index1 as f64, index1 as f64, 1.0)];
            let seqs2 = &[af::Seq::new(i as f64, i as f64, 1.0)
                , af::Seq::new(0.0, 0.0, 1.0)
                , af::Seq::new(index2 as f64, index2 as f64, 1.0)];
            ar2 = af::assign_seq(&ar2, seqs1, &one);
            ar2 = af::assign_seq(&ar2, seqs2, &one);
        }
        af::join(2, &ar1, &ar2)


    }

    fn generate_target(&self, input: &Array, batch_size: u64, bptt_unroll: u64) -> Array {
        let first = af::slices(&input, 0, bptt_unroll/2-1);
        let second = af::slices(&input, bptt_unroll/2, bptt_unroll-1);
        let zeros = af::constant(0. as f32, first.dims());
        let ones = af::constant(1. as f32, first.dims());
        let ar = af::mul(&first, &second, false);

        af::product(&af::select(&ar, &af::gt(&ar, &zeros, false), &ones), 2)



    }
}

impl DataSource for AddingProblemSource {
    fn get_train_iter(&self, num_batch: u64) -> Data {
        let inp = self.generate_input(num_batch
                                        , self.params.input_dims[2]);
        let tar = self.generate_target(&inp
                                       , num_batch
                                       , self.params.input_dims[2]);
        let mut batch = Data {
            input: RefCell::new(Box::new(inp.clone())),
            target: RefCell::new(Box::new(tar.clone())),
        };
        
        let current_iter = self.params.current_epoch.get();
        if self.iter.get() == self.params.num_samples as u64/ num_batch as u64 {
            self.params.current_epoch.set(current_iter + 1);
            self.iter.set(0);
        }
        self.iter.set(self.iter.get() + 1);
        batch
    }

    fn info(&self) -> DataParams {
        self.params.clone()
    }

    fn get_test_iter(&self, num_batch: u64) -> Data {
        self.get_train_iter(num_batch)
    }

    fn get_validation_iter(&self, num_batch: u64) -> Option<Data> {
       Some( self.get_train_iter(num_batch))
    }
}














