extern crate rand;
use rand::distributions::{Range, IndependentSample};
use af;
use af::{Array, Dim4, DType};
use std::cell::{RefCell, Cell};

use initializations;

use data::{Data, DataSource, DataParams, Normalize, Shuffle};

pub struct CopyingProblemSource {
    pub params: DataParams,
    pub iter: Cell<u64>,
    pub offset: Cell<f32>,
    pub bptt_unroll : u64, 
    pub seq_size : u64,
}

impl CopyingProblemSource {
    pub fn new(input_size: u64, batch_size: u64, seq_size: u64
               , bptt_unroll: u64, dtype: DType, max_samples: u64) -> CopyingProblemSource
    {
        let input_dims = Dim4::new(&[batch_size, input_size, bptt_unroll, 1]);
        //let output_dims = Dim4::new(&[batch_size, output_size, bptt_unroll, 1]);
        let train_samples = 0.7 * max_samples as f32;
        let test_samples = 0.2 * max_samples as f32;
        let validation_samples = 0.1 * max_samples as f32;
        CopyingProblemSource {
            params : DataParams {
                input_dims: input_dims,
                target_dims: input_dims,
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
            seq_size: seq_size,
        }
    }

    fn generate_input(&self, batch_size: u64, input_size: u64, bptt_unroll: u64, seq_size: u64) -> Array {

        let between = Range::new(0,input_size-2);
        let mut rng = rand::thread_rng();
        let mut ar1 = af::constant(0f32, af::Dim4::new(&[batch_size,input_size,seq_size,1]));
        let mut ar2 = af::constant(0f32, af::Dim4::new(&[batch_size,input_size,bptt_unroll-2*seq_size,1]));
        let mut ar3 = af::constant(0f32, af::Dim4::new(&[batch_size,input_size,seq_size,1]));
        let one = af::constant(1f32, af::Dim4::new(&[1,1,1,1]));

        for i in 0..seq_size {
            for j in 0..batch_size {
                let index = between.ind_sample(&mut rng);
                let seqs1 = &[af::Seq::new(j as f64,j as f64,1.0)
                    , af::Seq::new(index as f64,index as f64,1.0)
                    , af::Seq::new(i as f64,i as f64,1.0)];
                ar1 = af::assign_seq(&ar1, seqs1, &one);

                let seqs3 = &[af::Seq::new(j as f64,j as f64,1.0)
                    , af::Seq::new((input_size-2) as f64,(input_size-2) as f64,1.0)
                    , af::Seq::new(i as f64,i as f64,1.0)];
                ar3 = af::assign_seq(&ar3, seqs3, &one);
            }
        }
        for i in 0..(bptt_unroll-2*seq_size-1) {
            for j in 0..batch_size {
                let seqs2 = &[af::Seq::new(j as f64,j as f64,1.0)
                    , af::Seq::new((input_size-2) as f64, (input_size-2) as f64,1.0)
                    , af::Seq::new(i as f64,i as f64,1.0)];
                ar2 = af::assign_seq(&ar2, seqs2, &one);
            }
        }

        let seq = &[af::Seq::new(0.0,(batch_size-1) as f64, 1.0)
            , af::Seq::new((input_size-1) as f64,(input_size-1) as f64,1.0)
            , af::Seq::new((bptt_unroll-2*seq_size-1) as f64, (bptt_unroll-2*seq_size-1) as f64, 1.0)];
        ar2 = af::assign_seq(&ar2, seq, &one);

        af::join_many(2, vec!(&ar1,&ar2,&ar3))    
    }

    fn generate_target(&self, input: &Array, batch_size: u64, input_size: u64, bptt_unroll: u64, seq_size: u64) -> Array {
        let mut first = af::constant(0f32, af::Dim4::new(&[batch_size, input_size, bptt_unroll-seq_size,1]));
        let ones = af::constant(1f32, af::Dim4::new(&[batch_size, 1, 1, 1]));

        for i in 0..bptt_unroll-seq_size {
            first = af::set_slice(&first, &af::set_col(&af::slice(&first, i), &ones, input_size-2), i);
        }
        let second = af::slices(&input, 0, seq_size-1);
        af::join(2, &first, &second)
    }
}

impl DataSource for CopyingProblemSource {
    fn get_train_iter(&self, num_batch: u64) -> Data {
        let inp = self.generate_input(num_batch
                                        , self.params.input_dims[1]
                                        , self.params.input_dims[2]
                                        , self.seq_size);
        let tar = self.generate_target(&inp
                                       , num_batch
                                       , self.params.target_dims[1]
                                       , self.params.target_dims[2]
                                       , self.seq_size);
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














