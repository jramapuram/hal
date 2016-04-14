use af;
use std;
use csv;
use rand;
use rand::Rng;
use std::path::Path;
use std::ops::{Sub, Div};
use std::{str, cmp};
use std::cmp::Ordering;
use std::io::{Read, Write};
use tar::Archive;
use flate2::read::GzDecoder;
use flate2::GzHeader;
use std::fs::File;
use num::traits::Float;
use statistical::{standard_deviation, mean};
use af::{Dim4, Array, Aftype, Seq, AfError, HasAfEnum};
use itertools::Zip;
use rustc_serialize::Encodable;

use hyper::Client;
use hyper::header::Connection;


use error::HALError;

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

#[derive(PartialEq,PartialOrd)]
struct NonNan(f64);

impl NonNan {
  fn new(val: f64) -> Option<NonNan> {
    if val.is_nan() {
      None
    } else {
      Some(NonNan(val))
    }
  }
}

impl Eq for NonNan {}

impl Ord for NonNan {
  fn cmp(&self, other: &NonNan) -> Ordering {
    self.partial_cmp(other).unwrap()
  }
}

// Convert a vector of elements to a vector of Array
pub fn vec_to_array<T: HasAfEnum>(vec_values: Vec<T>, rows: usize, cols: usize) -> Array {
  raw_to_array(vec_values.as_ref(), rows, cols)
}

// Convert a generic vector to an Array
pub fn raw_to_array<T: HasAfEnum>(raw_values: &[T], rows: usize, cols: usize) -> Array {
  let dims = Dim4::new(&[rows as u64, cols as u64, 1, 1]);
  Array::new::<T>(raw_values, dims).unwrap()
}

// convert an array into a vector of rows
pub fn array_to_rows(input: &Array) -> Vec<Array> {
  let mut rows = Vec::new();
  for r in 0..input.dims().unwrap()[0] {
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
  for row in 0..row_count {
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
// SLOOOOOOW
pub fn shuffle_array(v: &mut[&mut Array], rows: u64) {
  let mut rng = rand::thread_rng();
  for row in 0..rows {
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
  match input.dims().unwrap().ndims() {
    4 => af::assign_seq(input, &[Seq::new(plane_num as f64, plane_num as f64, 1.0)
                                 , Seq::default()
                                 , Seq::default()
                                 , Seq::default()]
                        , new_plane),
    3 => af::assign_seq(input, &[Seq::new(plane_num as f64, plane_num as f64, 1.0)
                                 , Seq::default()
                                 , Seq::default()]
                        , new_plane),
    2 => af::assign_seq(input, &[Seq::new(plane_num as f64, plane_num as f64, 1.0)
                                 , Seq::default()]
                        , new_plane),
    1 => af::assign_seq(input, &[Seq::new(plane_num as f64, plane_num as f64, 1.0)]
                        , new_plane),
    _ => panic!("unknown dimensions provided to set_row_planes"),
  }
}

pub fn row_planes(input: &Array, first: u64, last: u64) -> Result<Array, AfError> {
  af::index(input, &[Seq::new(first as f64, last as f64, 1.0)
                     , Seq::default()
                     , Seq::default()])
}

pub fn set_row_planes(input: &Array, new_planes: &Array
                      , first: u64, last: u64) -> Result<Array, AfError>
{
  match input.dims().unwrap().ndims() {
    4 => af::assign_seq(input, &[Seq::new(first as f64, last as f64, 1.0)
                                 , Seq::default()
                                 , Seq::default()
                                 , Seq::default()]
                        , new_planes),
    3 => af::assign_seq(input, &[Seq::new(first as f64, last as f64, 1.0)
                                 , Seq::default()
                                 , Seq::default()]
                        , new_planes),
    2 => af::assign_seq(input, &[Seq::new(first as f64, last as f64, 1.0)
                                 , Seq::default()]
                        , new_planes),
    1 => af::assign_seq(input, &[Seq::new(first as f64, last as f64, 1.0)]
                        , new_planes),
    _ => panic!("unknown dimensions provided to set_row_planes"),
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

fn _read_gzip_filename(entire_file: &Vec<u8>) -> String {
  let d = match GzDecoder::new(&entire_file[..]) {
    Err(e) => panic!("Could not read gzip header: {}", e),
    Ok(dc) => dc,
  };
  str::from_utf8(d.header().filename().unwrap())
    .unwrap().to_owned()
}

fn _ungzip_to_file(dest: &str, entire_file: &Vec<u8>) -> bool{
  let mut d = match GzDecoder::new(&entire_file[..]) {
    Err(e) => panic!("Could not read gzip {}: {}", dest, e),
    Ok(dc) => dc,
  };

  let mut body = Vec::new();
  d.read_to_end(&mut body);
  let mut f = match File::create(dest){
    Err(e) => panic!("cannot create file {}", dest),
    Ok(f)  => f,
  };
  f.write_all(&body[..]);
  true
}

pub fn ungzip(src: &str) {
  let mut file = match File::open(src){
    Err(e) => panic!("could not open {}, {}", src, e),
    Ok(f)  => f,
  };
  let mut entire_file = Vec::new();
  file.read_to_end(&mut entire_file);

  // get the filename
  let name = _read_gzip_filename(&entire_file);
  print!("ungzip'ing {} from {}...", name, src);

  // write to dest
  _ungzip_to_file(&name, &entire_file);
  println!("...completed");
}

pub fn untar(src: &str, dest: &str){
  print!("Untarring {}...", src);
  let mut ar = Archive::new(File::open(src).unwrap());
  ar.unpack(dest).unwrap();
  println!("...complete");
}

/// Pull a file from URL to a location destination
///
/// # Parameters
/// - `url` is the location to pull data from
/// - `dest` is the destination file location
pub fn download(url: &str, dest: &str) {
  print!("Downloading {} to {}...", url, dest);
  let mut client = Client::new();
  let mut res = client.get(url)
                      .header(Connection::close())
                      .send().unwrap();
  let mut body = Vec::new();
  res.read_to_end(&mut body).unwrap();

  let mut f = match File::create(dest){
    Err(e) => panic!("cannot open file {}", dest),
    Ok(f)  => f,
  };
  f.write_all(&body[..]);
  println!("...complete: read {} Mb"
           , body.len() as f32/(1024.0*1024.0));
}

/// Returns true if the file exists (and is a file)
pub fn file_exists(path: &str) -> bool{
  let path = Path::new(path);
  path.exists() & path.is_file()
}

/// Returns true if the directory exists (and is a dir)
pub fn dir_exists(path: &str) -> bool{
  let path = Path::new(path);
  path.exists() & path.is_dir()
}

/// Gradient checking helper for smooth functions like :
/// tanh / sigmoid / softmax
///
/// # Parameters
/// - `F` is the function whose gradient we will evaluate
/// - `input` is the input data array
/// - `eps` is a very small number (Generally 1e-5)
/// - `grad` is your evaluated gradient
pub fn verify_gradient_smooth<F>(fn_closure: F, input: &Array, eps: f64, grad: &Array) -> Result<f64, HALError>
  where F : Fn(&Array) -> Array
{
  let rel = gradient_check(fn_closure, input, eps, grad);
  println!("Relative error = {}", rel);
  match rel {
    n if n > 1e-2             => Err(HALError::GRADIENT_ERROR),
    n if n < 1e-2 && n > 1e-4 => Err(HALError::GRADIENT_ERROR),
    _                         => Ok(rel),
  }
}

/// Gradient checking helper for objective with 'kinks' like:
/// relu, lrelu, etc
///
/// # Parameters
/// - `F` is the function whose gradient we will evaluate
/// - `input` is the input data array
/// - `eps` is a very small number (Generally 1e-5)
/// - `grad` is your evaluated gradient
pub fn verify_gradient_kinks<F>(fn_closure: F, input: &Array, eps: f64, grad: &Array) -> Result<f64, HALError>
  where F : Fn(&Array) -> Array
{
  let rel = gradient_check(fn_closure, input, eps, grad);
  println!("Relative error = {}", rel);
  match rel {
    n if n > 1e-2 => Err(HALError::GRADIENT_ERROR),
    n if n < 1e-4 => Ok(rel),
    _             => Ok(rel),
  }
}

/// Gradient checking helper that accepts perturbations
///
/// df(input)/dinput = ((f(input + eps)- f(input - eps))/(2*eps)
/// everything is tabulated in double precision
///
/// # Parameters
/// - `fi_plus_eps` is f(input + eps)
/// - `fi_minus_eps` is f(input - eps)
/// - `input` is the input data array
/// - `eps` is a very small number (Generally 1e-5)
/// - `grad` is your evaluated gradient
pub fn gradient_check_with_perturbations(fi_plus_eps: &Array, fi_minus_eps: &Array, eps: f64, grad: &Array) -> f64
{
  let num_grad = af::div(&af::sub(fi_plus_eps, fi_minus_eps, false).unwrap()
                         , &(2.0f64 * eps), false).unwrap();

  // now calculate the relative error
  let abs_diff_grads = af::abs(&af::sub(&grad.cast::<f64>().unwrap(), &num_grad, false).unwrap()).unwrap();
  let abs_num_grad = NonNan::new(af::sum_all(&af::abs(&num_grad).unwrap()).unwrap().0).unwrap();
  let abs_grad = NonNan::new(af::sum_all(&af::abs(&grad.cast::<f64>().unwrap()).unwrap()).unwrap().0).unwrap();
  af::sum_all(&abs_diff_grads).unwrap().0 / cmp::max(abs_grad, abs_num_grad).0
}


/// Gradient checking helper
///
/// df(input)/dinput = ((f(input + eps)- f(input - eps))/(2*eps)
/// everything is tabulated in double precision
///
/// # Parameters
/// - `F` is the function whose gradient we will evaluate
/// - `input` is the input data array
/// - `eps` is a very small number (Generally 1e-5)
/// - `grad` is your evaluated gradient
pub fn gradient_check<F>(fn_closure: F, input: &Array, eps: f64, grad: &Array) -> f64
  where F : Fn(&Array) -> Array
{
  // calculate the numerical gradient
  let fi_plus_eps = fn_closure(&af::add(&input.cast::<f64>().unwrap(), &eps, false).unwrap());
  let fi_minus_eps = fn_closure(&af::sub(&input.cast::<f64>().unwrap(), &eps, false).unwrap());
  // assert!(f_input_minus_eps.get_type().unwrap() == f_input_plus_eps.get_type().unwrap(),
  //         "Gradient checking failed to typecast input to a double array");
  gradient_check_with_perturbations(&fi_plus_eps, &fi_minus_eps, eps, grad)
}
