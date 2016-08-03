use af;
use std;
use csv;
use rand;
use rand::Rng;
use conv::{ConvUtil, Saturate};
//use conv::errors::GeneralErrorKind;
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
use num::{Complex, Num};
use statistical::{standard_deviation, mean};
use af::{Dim4, Array, DType, Seq, HasAfEnum};
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

/// Helper to assert all types in a vector are the same
pub fn assert_types(v: Vec<&Array>){
  let base_type = v[0].get_type();
  for i in 1..v.len() {
    let cur_type = v[i].get_type();
    assert!(cur_type == base_type
            , "type mismatch detected: {:?} vs {:?}"
            , cur_type, base_type);
  }
}

/// Helper to return a constant value based on type
pub fn constant(dims: Dim4, aftype: DType, val: f32) -> Array {
  match aftype
  {
    DType::F32 => af::constant(val, dims),
    DType::F64 => af::constant(val.approx_as::<f64>().unwrap(), dims),
    DType::C32 => af::constant(Complex::new(val, 0f32)
                                , dims),
    DType::C64 => af::constant(Complex::new(val.approx_as::<f64>().unwrap(), 0f64)
                        , dims),
    DType::B8  => {
      if val > 0f32 {
        af::constant(true, dims)
      }else{
        af::constant(false, dims)
      }
    },
    DType::S32 => af::constant(val.approx_as::<i32>().saturate().unwrap(), dims),
    DType::U32 => af::constant(val.approx_as::<u32>().saturate().unwrap(), dims),
    DType::U8  => af::constant(val.approx_as::<u8>().saturate().unwrap(), dims),
    DType::S64 => af::constant(val.approx_as::<i64>().saturate().unwrap(), dims),
    DType::U64 => af::constant(val.approx_as::<u64>().saturate().unwrap(), dims),
    DType::S16 => af::constant(val.approx_as::<i16>().saturate().unwrap(), dims),
    DType::U16 => af::constant(val.approx_as::<u16>().saturate().unwrap(), dims),
  }
}

pub fn cast(input: &Array, dest_type: DType) -> Array {
  if input.get_type() == dest_type{
    return input.clone()
  }

  match dest_type
  {
    DType::F32 => input.cast::<f32>(),
    DType::F64 => input.cast::<f64>(),
    DType::C32 => input.cast::<Complex<f32>>(),
    DType::C64 => input.cast::<Complex<f64>>(),
    DType::B8  => input.cast::<bool>(),
    DType::S32 => input.cast::<i32>(),
    DType::U32 => input.cast::<u32>(),
    DType::U8  => input.cast::<u8>(),
    DType::S64 => input.cast::<i64>(),
    DType::U64 => input.cast::<u64>(),
    DType::S16 => input.cast::<i64>(),
    DType::U16 => input.cast::<u16>(),
  }
}


/// Convert a vector of elements to a vector of Array
pub fn vec_to_array<T: HasAfEnum>(vec_values: Vec<T>, dims: Dim4) -> Array {
  raw_to_array(vec_values.as_ref(), dims)
}

/// Convert a generic vector to an Array
pub fn raw_to_array<T: HasAfEnum>(raw_values: &[T], dims: Dim4) -> Array {
  Array::new::<T>(raw_values, dims)
}

/// convert an array into a vector of rows
pub fn array_to_rows(input: &Array) -> Vec<Array> {
  let mut rows = Vec::new();
  for r in 0..input.dims()[0] {
    rows.push(af::row(input, r as u64));
  }
  rows
}

/// convery an array to a single vector [loses dimensions]
pub fn array_to_vec(input: &Array) -> Vec<f64>
{
  let elems = input.dims().elements();
  let mut v: Vec<f64> = vec![0f64; elems as usize];
  input.host(&mut v);
  v
}

// convert a vector of rows into a single array
pub fn rows_to_array(input: Vec<&Array>) -> Array {
  // let mut arr = vec![input[0]];
  // // af_join_many supports up to 10 (9 + previous) arrays being joined at once
  // for rows in input[1..input.len()].iter().collect::<Vec<_>>().chunks(9) {
  //   arr.extend(Vec::from(rows));
  //   arr = vec![&af::join_many(0, arr)];
  // }
  // arr[0].clone();
  if input.len() > 10 {
    panic!("cannot currently handle array merge of more than 10 items");
  }

  af::join_many(0, input)
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
      let dims = mat.dims();
      let rnd_plane  = row_plane(mat, rnd_row);
      let orig_plane = row_plane(mat, dims[0] - row - 1);
      **mat = set_row_plane(mat, &rnd_plane, dims[0] - row - 1);
      **mat = set_row_plane(mat, &orig_plane, rnd_row);
    }
  }
}

pub fn row_plane(input: &Array, slice_num: u64) -> Array {
  af::index(input, &[Seq::new(slice_num as f64, slice_num as f64, 1.0)
                     , Seq::default()
                     , Seq::default()])
}

pub fn set_row_plane(input: &Array, new_plane: &Array, plane_num: u64) -> Array {
  match input.dims().ndims() {
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

pub fn row_planes(input: &Array, first: u64, last: u64) -> Array {
  af::index(input, &[Seq::new(first as f64, last as f64, 1.0)
                     , Seq::default()
                     , Seq::default()])
}

pub fn set_row_planes(input: &Array, new_planes: &Array
                      , first: u64, last: u64) -> Array
{
  match input.dims().ndims() {
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

pub fn is_nan(input: &Array) -> bool {
  let nan_array = af::isnan(&input);
  return af::sum_all(&nan_array).0 > 0f64
}

pub fn clip_by_value(src: &Array, clip_min: f32, clip_max: f32) -> Array {
  let min_clipped = af::selectl(clip_min as f64, &af::lt(src, &clip_min, false), src);
  af::selectl(clip_max as f64, &af::gt(&min_clipped, &clip_max, false), &min_clipped)
}

// Normalize an array based on mean & num_std_dev deviations of the variance
pub fn normalize_array(src: &Array, num_std_dev: f32) -> Array {
  let mean = af::mean_all(src).0 as f32;
  let mut std_dev = num_std_dev * (af::var_all(src, false).0 as f32).sqrt().abs();
  std_dev = std_dev + 1e-9f32; // to not divide by zero
  af::div(&af::sub(src, &mean, false), &std_dev, false)
}

pub fn scale(src: &Array, low: f32, high: f32) -> Array {
  let min = af::min_all(&src).0 as f32;
  let max = af::max_all(&src).0 as f32;

  af::add(&af::div(&af::mul(&(high - low), &af::sub(src, &min, false), false)
                   , &(max - min), false)
          , &low, false)
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
  d.read_to_end(&mut body).unwrap();
  let mut f = match File::create(dest){
    Err(e) => panic!("cannot create file {}: {}", dest, e),
    Ok(f)  => f,
  };
  f.write_all(&body[..]).unwrap();
  true
}

pub fn ungzip(src: &str) {
  let mut file = match File::open(src){
    Err(e) => panic!("could not open {}, {}", src, e),
    Ok(f)  => f,
  };
  let mut entire_file = Vec::new();
  file.read_to_end(&mut entire_file).unwrap();

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
  let client = Client::new();
  let mut res = client.get(url)
                      .header(Connection::close())
                      .send().unwrap();
  let mut body = Vec::new();
  res.read_to_end(&mut body).unwrap();

  let mut f = match File::create(dest){
    Err(e) => panic!("cannot open file {}: {}", dest, e),
    Ok(f)  => f,
  };
  f.write_all(&body[..]).unwrap();
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
  where F : Fn(&Array) -> f64
{
  let rel = gradient_check(fn_closure, input, eps, grad);
  println!("Relative error[smooth] = {}", rel);
  match rel {
    n if n > 1e-2 => Err(HALError::GRADIENT_ERROR),
    n if n < 1e-5 => Err(HALError::GRADIENT_ERROR),
    _             => Ok(rel),
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
  where F : Fn(&Array) -> f64
{
  let rel = gradient_check(fn_closure, input, eps, grad);
  println!("Relative error[kinks] = {}", rel);
  match rel {
    n if n > 1e-2 => Err(HALError::GRADIENT_ERROR),
    n if n < 1e-4 => Ok(rel),
    _             => Ok(rel),
  }
}

/// Numerical gradient calculator
///
/// df(input)/dinput = ((f(input + eps)- f(input - eps))/(2*eps)
///
/// # Parameters
/// - `F` is a closure that returns a pointwise derivative
/// - `arr` is the input data array
/// - `eps` is a very small number (Generally 1e-5)
pub fn numerical_gradient<F>(fn_closure: F, arr: &Array, eps: f64) -> Array
  where F : Fn(&Array) -> f64
{
  // build a vector to hold the gradients
  let num_elems = arr.elements() as usize;
  let num_rows = arr.dims()[0] as usize;
  let num_cols = arr.dims()[1] as usize;
  let mut grad_vec = vec![0f64; num_elems];

  let target_dims = Dim4::new(&[num_rows as u64, num_cols as u64, 1, 1]);
  for i in 0..num_elems
  {
    // build a vec(matrix) of [0....eps...0] (same size as arr)
    // then add and subtract it correspondingly.
    let mut eps_vec = vec![0f64; num_elems];
    eps_vec[i] = eps;

    let eps_arr = vec_to_array::<f64>(eps_vec.clone(), target_dims);
    let arr_p_h = af::add(arr, &eps_arr, false);
    let arr_m_h = af::sub(arr, &eps_arr, false);

    // run it through the function to map to R1
    grad_vec[i] = (fn_closure(&arr_p_h) - fn_closure(&arr_m_h)) / (2f64 * eps);
  }

  vec_to_array::<f64>(grad_vec, target_dims)
}


/// Gradient checking helper
///
/// rel = abs(grad - numgrad) / max(abs(grad), abs(numgrad))
///
/// # Parameters
/// - `F` is a closure that returns a pointwise derivative
/// - `input` is the input data array
/// - `eps` is a very small number (Generally 1e-5)
/// - `grad` is your analytically evaluated gradient
pub fn gradient_check<F>(fn_closure: F, input: &Array, eps: f64, grad: &Array) -> f64
  where F : Fn(&Array) -> f64
{
  // calculate the numerical gradient
  let num_grad = numerical_gradient(fn_closure, input, eps);
  println!("numgrad = {:?}\ngrad = {:?}", array_to_vec(&num_grad), array_to_vec(grad));

  // now calculate the relative error
  let abs_diff_grads = af::abs(&af::sub(&grad.cast::<f64>(), &num_grad, false));
  let abs_num_grad = NonNan::new(af::sum_all(&af::abs(&num_grad)).0).unwrap();
  let abs_grad = NonNan::new(af::sum_all(&af::abs(&grad.cast::<f64>())).0).unwrap();
  println!("abs tot = {}  | ag = {} | ang = {}", af::sum_all(&abs_diff_grads).0, abs_grad.0, abs_num_grad.0);
  af::sum_all(&abs_diff_grads).0 / cmp::max(abs_grad, abs_num_grad).0
}
