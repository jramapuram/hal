use af;
use af::Array;

use utils;
//use error::HALError;

pub fn plot_array(values: &Array, title: &str, window_x: u16, window_y: u16) {
  assert!(values.dims().unwrap()[1] == 1);

  // create a window
  let title_str = String::from(title);
  let wnd = match af::Window::new(window_x as i32, window_y as i32, title_str) {
    Ok(v)  => v,
    Err(e) => panic!("Window creation failed: {}", e), //XXX: handle better
  };

  // display till closed
  loop {
    wnd.draw_plot(&af::range(values.dims().unwrap().clone()
                             , 0, af::Aftype::F32).unwrap(), &values, None);
    if wnd.is_closed().unwrap() == true { break; }
  }
}

pub fn plot_vec<T>(raw_values: Vec<T>, title: &str, window_x: u16, window_y: u16) {
  // copy from float vector to Array
  let num_rows = raw_values.len();
  let values = utils::vec_to_array(raw_values, num_rows, 1);
  plot_array(&values, title, window_x, window_y);
}
