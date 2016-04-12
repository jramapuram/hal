use af::Array;
use std::fs::{create_dir_all, File};
use std::cell::{RefCell, Cell};
use tar::Archive;

use hal::utils;
use hal::data::{Data, DataSource, DataParams, Normalize, Shuffle};

pub struct MNISTSource {
  pub params: DataParams,
  pub iter: Cell<u64>,
}

impl MNISTSource {
  fn new(batch_size: u64, is_normalized: bool, is_shuffled: bool) -> MNISTSource
  {
    let base_url = "http://yann.lecun.com/exdb/mnist/";
    let train_images_file = "train-images-idx3-ubyte.gz";
    let train_labels_file = "train-labels-idx1-ubyte.gz";
    let test_images_file = "t10k-images-idx3-ubyte.gz";
    let test_labels_file = "t10k-labels-idx1-ubyte.gz";

    // pull our data if we don't have it already
    download_and_extract(base_url, train_images_file, train_labels_file
                         , test_images_file, test_labels_file);

    let validation_size = 5000;
    let input_size = 784;
    let target_size = 10;

    let idims = Dim4::new(&[batch_size, input_size, 1, 1]);
    let odims = Dim4::new(&[batch_size, target_size, 1, 1]);
    let train_samples = 0.7 * max_samples as f32;
    let test_samples = 0.2 * max_samples as f32;
    let validation_samples = 0.1 * max_samples as f32;

    MNISTSource {
      params: DataParams {
        input_dims: idims,
        target_dims: odims,
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
    }
  }

  fn verify_or_download(dir_path: &str, filename: &str){
    if !utils::dir_exists(dir_path) {
      create_dir_all(dir_path).unwrap();
    }
    // download and extract images & labels if they dont exist
    let fq_filename = &format!("{}{}", dir_path, filename)
    if !utils::file_exists(fq_filename){
      utils::download(&format!("{}{}", base_url, train_images_file), train_images_file);
      utils::ungzip(train_images_file);
    }
  }

  fn download_and_extract(base_url: &str, train_images_file: &str
                          , train_labels_file: &str, test_images_file: &str
                          , test_labels_file: &str)
  {
    // download and extract the training data
    verify_or_download(base_url, train_images_file);
    verify_or_download(base_url, train_labels_file);

    // download and extract the test data
    verify_or_download(base_url, test_images_file);
    verify_or_download(base_url, test_labels_file);
  }
}

impl DataSource for SinSource
{
  fn get_train_iter(&self, num_batch: u64) -> Data {
    let inp = self.generate_sin_wave(self.params.input_dims[1], num_batch);
    let mut batch = Data {
      input: RefCell::new(Box::new(inp.clone())),
      target: RefCell::new(Box::new(inp.copy().unwrap())),
    };

    if self.params.normalize { batch.normalize(3.0); }
    if self.params.shuffle   {  batch.shuffle(); }
    let current_iter = self.params.current_epoch.get();
    if self.iter.get()  == self.params.num_samples as u64/ num_batch as u64 {
      self.params.current_epoch.set(current_iter + 1);
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
    Some(self.get_train_iter(num_batch))
  }
}
