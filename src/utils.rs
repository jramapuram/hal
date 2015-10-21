use af;
use std;
use csv;
use rand;
use rand::Rng;
use std::path::Path;
use std::ops::Sub;
use num::traits::Float;
use statistical::{standard_deviation, mean};
use af::{Dim4, Array, Aftype, AfBackend, Seq, AfError};
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

// convert an array into a vector of rows
pub fn array_to_rows(input: &Array) -> Vec<Array> {
  let mut rows = Vec::new();
  for r in (0..input.dims().unwrap()[0]) {
    rows.push(af::row(input, r as u64).unwrap());
  }
  rows
}

// convert a vector of rows into a single array
pub fn rows_to_array(input: Vec<&Array>) -> Array {
  // let mut arr = vec![input[0]];
  // // af_join_many supports up to 10 (9 + previous) arrays being joined at once
  // for rows in input[1..input.len()].iter().collect::<Vec<_>>().chunks(9) {
  //   arr.extend(Vec::from(rows));
  //   arr = vec![&af::join_many(0, arr).unwrap()];
  // }
  // arr[0].clone();
  if input.len() > 10 {
    panic!("cannot currently handle array merge of more than 10 items");
  }

  af::join_many(0, input).unwrap()
}

// Convert an array from one backend to the other
pub fn array_swap_backend(input: &Array
                          , backend: af::AfBackend
                          , device_id: i32) -> Array
{
  let dims = input.dims().unwrap();
  let mut buffer = vec![0f32; dims.elements() as usize];
  input.host(&mut buffer).unwrap();
  af::set_backend(backend).unwrap();
  af::set_device(device_id).unwrap();
  Array::new(dims, &buffer, Aftype::F32).unwrap()
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

// Randomly shuffle planes of an array
pub fn shuffle_array(v: &mut[&mut Array], rows: u64) {
  let mut rng = rand::thread_rng();
  for row in (0..rows) {
    let rnd_row = rng.gen_range(0, rows - row);
    for mat in v.iter_mut() { //swap all tensors similarly
      let dims = mat.dims().unwrap();
      let rnd_plane  = row_plane(mat, rnd_row).unwrap();
      let orig_plane = row_plane(mat, dims[0] - row - 1).unwrap();
      **mat = set_row_plane(mat, &rnd_plane, dims[0] - row - 1).unwrap();
      **mat = set_row_plane(mat, &orig_plane, rnd_row).unwrap();
    }
  }
}

pub fn row_plane(input: &Array, slice_num: u64) -> Result<Array, AfError> {
  af::index(input, &[Seq::new(slice_num as f64, slice_num as f64, 1.0)
                     , Seq::default()
                     , Seq::default()])
}

pub fn set_row_plane(input: &Array, new_plane: &Array, plane_num: u64) -> Result<Array, AfError> {
  af::assign_seq(input, &[Seq::new(plane_num as f64, plane_num as f64, 1.0)
                          , Seq::default(), Seq::default()]
                 , new_plane)
}

pub fn row_planes(input: &Array, first: u64, last: u64) -> Result<Array, AfError> {
  af::index(input, &[Seq::new(first as f64, last as f64, 1.0)
                     , Seq::default()
                     , Seq::default()])
}

pub fn set_row_planes(input: &Array, new_planes: &Array, first: u64, last: u64) -> Result<Array, AfError> {
  af::assign_seq(input, &[Seq::new(first as f64, last as f64, 1.0)
                          , Seq::default()
                          , Seq::default()]
                 , new_planes)
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
