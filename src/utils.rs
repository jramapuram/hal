extern crate rand;

use af;
use af::{Dim4, Array, Aftype};
use na::DMat;
use rand::Rng;

use error::HALError;

// Convert a vector of elements to a vector of Array
pub fn vec_to_array<T>(vec_values: Vec<T>, rows: usize, cols: usize) -> Array {
  raw_to_array(vec_values.as_ref(), rows, cols)
}

// Convert a generic vector to an Array
pub fn raw_to_array<T>(raw_values: &[T], rows: usize, cols: usize) -> Array {
  let dims = Dim4::new(&[rows as u64, cols as u64, 1, 1]);
  &Array::new(dims, &raw_values, Aftype::F32).unwrap()
}

// Convert a dmat of elements to an array
pub fn dmat_to_array<T>(dmat_values: &DMat<T>) -> Array {
  raw_to_array(dmat_values.as_vec(), dmat_values.shape().0, dmat_values.shape().1)
}

// Convert a GPU array to a dmat
pub fn array_to_dmat<T>(arr: &Array) -> DMat<T>{
  let mut retval = DMat::<T>::new_zeros(arr.dims[0], arr.dims[1]);
  af::host(retval.as_mut_vec()).unwrap();
  retval
}

// Helper to swap rows (row major order) in a generic type [non GPU]
fn swap_rowR<T>(matrix: &mut [T], row_src: usize, row_dest: usize, cols: usize){
  assert!(matrix.len() % cols == 0);
  if row_src != row_dest {
    for c in 0..cols {
      matrix.swap(cols * row_src + c, cols * row_dest + c);
    }
  }
}

// Helper to swap rows (col major order) in a generic type [non GPU]
fn swap_rowC<T>(matrix: &mut [T], row_src: usize, row_dest: usize, cols: usize){
  assert!(matrix.len() % cols == 0);
  let row_count = matrix.len()/cols;
  if row_src != row_dest {
    for c in 0..cols {
      matrix.swap(c * row_count + row_src, c * row_count + row_dest);
    }
  }
}

// Randomly shuffle a set of 2d matrices [or vectors] using knuth shuffle
fn shuffle<T>(v: &mut[&mut [T]], cols: [usize], row_major: bool) {
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
        true  => swap_rowR(mat, rnd_row, row_count - row - 1, col),
        false => swap_rowC(mat, rnd_row, row_count - row - 1, col),
      };
    }
  }
}

// A helper to return a batching iterator
fn batch<T>(v: &[T], batch_size: usize){
  v[..].chunks(batch_size)
}

// Normalize an array based on mean & num_std_dev deviations of the variance
fn normalize(src: &Array, num_std_dev: f32) -> Array {
  let mean = arr.mean_all().0;
  let var = arr.var_all.0;
  af::div(af::sub(&arr, mean).unwrap(), af::mul(num_std_dev, var))
}
