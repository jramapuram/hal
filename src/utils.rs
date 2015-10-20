use af;
use std;
use csv;
use rand;
use rand::Rng;
use std::path::Path;
use std::ops::Sub;
use num::traits::Float;
use statistical::{standard_deviation, mean};
use af::{Dim4, Array, Aftype};
use na::{DMat, DVec, Shape};
use itertools::Zip;
use rustc_serialize::Encodable;

//use error::HALError;

// allows for let a = hashmap!['key1' => value1, ...];
// http://stackoverflow.com/questions/28392008/more-concise-hashmap-initialization
#[macro_export]
macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

// Convert a vector of elements to a vector of Array
pub fn vec_to_array<T>(vec_values: Vec<T>, rows: usize, cols: usize) -> Array {
  raw_to_array(vec_values.as_ref(), rows, cols)
}

// Convert a generic vector to an Array
pub fn raw_to_array<T>(raw_values: &[T], rows: usize, cols: usize) -> Array {
  let dims = Dim4::new(&[rows as u64, cols as u64, 1, 1]);
  Array::new(dims, &raw_values, Aftype::F32).unwrap()
}

// Convert an array from one backend to the other
pub fn array_swap_backend(input: &Array
                          , backend: af::AfBackend
                          , device_id: u8) -> Array
{
  let mut buffer: [f32; input.dims().unwrap().elements()];
  input.host(&mut buffer);
  af::set_backend(backend).unwrap();
  af::set_device(device_id).unwrap();
  Array::new(input.dims, &buffer, Aftype::F32).unwrap()
}

// Helper to swap rows (row major order) in a generic type [non GPU]
pub fn swap_row<T>(matrix: &mut [T], row_src: usize, row_dest: usize, cols: usize){
  assert!(matrix.len() % cols == 0);
  if row_src != row_dest {
    for c in 0..cols {
      matrix.swap(cols * row_src + c, cols * row_dest + c);
    }
  }
}

// Helper to swap rows (col major order) in a generic type [non GPU]
pub fn swap_col<T>(matrix: &mut [T], row_src: usize, row_dest: usize, cols: usize){
  assert!(matrix.len() % cols == 0);
  let row_count = matrix.len() / cols;
  if row_src != row_dest {
    for c in 0..cols {
      matrix.swap(c * row_count + row_src, c * row_count + row_dest);
    }
  }
}

// Randomly shuffle a set of 2d matrices [or vectors] using knuth shuffle
pub fn shuffle_matrix<T>(v: &mut[&mut [T]], cols: &[usize], row_major: bool) {
  assert!(v.len() > 0 && cols.len() > 0);

  let total_length = v[0].len();
  assert!(total_length % cols[0] == 0);
  let row_count = total_length / cols[0];

  let mut rng = rand::thread_rng();
  for row in (0..row_count) {
    let rnd_row = rng.gen_range(0, row_count - row);
    for (mat, col) in Zip::new((v.iter_mut(), cols.iter())) { //swap all matrices similarly
      assert!(mat.len() % col == 0);
      match row_major{
        true  => swap_row(mat, rnd_row, row_count - row - 1, col.clone()),
        false => swap_col(mat, rnd_row, row_count - row - 1, col.clone()),
      };
    }
  }
}

pub fn row_plane(input: &Array, slice_num: u64) -> Result<Array, AfError> {
  assign_seq(input, &[Seq::new(slice_num as f64, slice_num as f64, 1.0)
                      , Seq::default(), Seq::default()], new_row)
}

pub fn set_row_plane(input: &Array, new_slice: &Array, slice_num: u64) -> Result<Array, AfError> {
  assign_seq(input, &[Seq::new(slice_num as f64, slice_num as f64, 1.0)]
             , Seq::default()
             , Seq::default()
             , new_slice)
}

// Randomly shuffle planes of an array
pub fn shuffle_array(o: &mut[&mut Array], rows: u64) {
  let mut rng = rand::thread_rng();
  for row in (0..rows) {
    let rnd_row = rng.gen_range(0, rows - row);
    for mat in v.iter_mut() { //swap all tensors similarly
      let dims = mat.dims().unwrap().get();
      let rnd_slice = row_plane(mat, rnd_row).unwrap();
      let orig = row_plane(mat, dims[0] - row - 1).unwrap();
      mat = set_row_plane(mat, rnd_slice, dims[0] - row - 1).unwrap();
      mat = set_row_plane(mat, orig_slice, rnd_row).unwrap();
    }
  }
}

// Helper to write a vector to a csv file
pub fn write_csv<T>(filename: &str, v: &Vec<T>)
  where T: Encodable
{
  let wtr = csv::Writer::from_file(Path::new(filename));
  match wtr {
    Ok(mut writer) => {
      for record in v {
        let result = writer.encode(record);
        assert!(result.is_ok());
      }
    },
    Err(e)    => panic!("error writing to csv file {} : {}", filename, e),
  };
}

// Helper to read a csv file to a vector
pub fn read_csv<T>(filename: &str) -> Vec<T>
  where T: std::str::FromStr, <T as std::str::FromStr>::Err: std::fmt::Debug
{
  let mut retval: Vec<T> = Vec::new();
  let rdr = csv::Reader::from_file(Path::new(filename));
  match rdr {
    Ok(mut reader) => {
      for row in reader.records() {
        let row = row.unwrap();
        for value in row {
          retval.push(value.parse::<T>().unwrap());
        }
      }
    },
    Err(e)     => panic!("error reader from csv file {} : {}", filename, e),
  }
  retval
}

// Generic Normalizer
pub fn normalize<T: Float + Sub>(src: &[T], num_std_dev: T) -> Vec<T> {
  let mean = mean(src);
  let std_dev = standard_deviation(src, Some(mean));
  src.iter().map(|&x| (x - mean) / (num_std_dev * std_dev)).collect()
}

// Normalize an array based on mean & num_std_dev deviations of the variance
pub fn normalize_array(src: &Array, num_std_dev: f32) -> Array {
  let mean = af::mean_all(src).unwrap().0 as f32;
  let var = num_std_dev * af::var_all(src, false).unwrap().0 as f32;
  if var > 0.00000001 || var < 0.00000001 {
    af::div(&af::sub(src, &mean, false).unwrap(), &var, false).unwrap()
  }else{
    af::sub(src, &mean, false).unwrap()
  }
}

pub fn scale(src: &Array, low: f32, high: f32) -> Array {
  let min = af::min_all(&src).unwrap().0 as f32;
  let max = af::max_all(&src).unwrap().0 as f32;

  af::add(&af::div(&af::mul(&(high - low), &af::sub(src, &min, false).unwrap(), false).unwrap()
                   , &(max - min), false).unwrap()
          , &low, false).unwrap()
}
