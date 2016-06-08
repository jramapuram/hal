use af;
use af::{Array, Dim4};
use std::cell::{RefCell, Cell};

use initializations;

use data::{Data, DataSource, DataParams, Normalize, Shuffle};

pub struct MultUniformSource {
    pub params: DataParams,
    pub iter: Cell<u64>,
    pub offset: Cell<f32>,
    pub bptt_unroll : u64, 
}

impl MultUniformSource {
    pub fn new(input_size: u64, output_size: u64, batch_size: u64
           , bptt_unroll: u64, max_samples: u64, is_normalized: bool
           , is_shuffled: bool) -> MultUniformSource
    {
        let input_dims = Dim4::new(&[batch_size, input_size, bptt_unroll, 1]);
        //let output_dims = Dim4::new(&[batch_size, output_size, bptt_unroll, 1]);
        let train_samples = 0.7 * max_samples as f32;
        let test_samples = 0.2 * max_samples as f32;
        let validation_samples = 0.1 * max_samples as f32;
        MultUniformSource {
            params : DataParams {
                input_dims: input_dims,
                target_dims: input_dims,
                normalize: is_normalized,
                shuffle: is_shuffled,
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

    fn generate_uniform(&self, batch_size: u64, input_size: u64, bptt_unroll: u64) -> Array {
        initializations::uniform::<f32>(Dim4::new(&[batch_size
                                                  , input_size
                                                  , bptt_unroll
                                                  , 1]), 0.05f32)
    }
}

impl DataSource for MultUniformSource {
    fn get_train_iter(&self, num_batch: u64) -> Data {
        let inp = self.generate_uniform(num_batch
                                        , self.params.input_dims[1]
                                        , self.params.input_dims[2]);
        let mut batch = Data {
            input: RefCell::new(Box::new(inp.clone())),
            target: RefCell::new(Box::new(inp.copy())),
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














