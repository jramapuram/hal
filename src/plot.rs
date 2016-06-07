use af;
use af::{Array, HasAfEnum};

use utils;
//use error::HALError;

pub fn plot_array(values: &Array, title: &str, window_x: u16, window_y: u16) {
  assert!(values.dims()[1] == 1);

  // create a window
  let title_str = String::from(title);
  let wnd = af::Window::new(window_x as i32, window_y as i32, title_str);

  // display till closed
  loop {
    wnd.draw_plot(&af::range::<f32>(values.dims().clone()
                                    , 0), &values, None);
    if wnd.is_closed() == true { break; }
  }
}

pub fn plot_vec<T: HasAfEnum>(raw_values: Vec<T>, title: &str, window_x: u16, window_y: u16) {
  // copy from float vector to Array
  let num_rows = raw_values.len();
  let values = utils::vec_to_array(raw_values, num_rows, 1);
  plot_array(&values, title, window_x, window_y);
}
