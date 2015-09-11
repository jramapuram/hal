use af;
use af::{Dim4, Array, Aftype};

use utils;
use error::HALError;

pub fn Plot<T>(raw_values: Vec<T>, title: &'static str, window_x: u16, window_y: u16) {
  // copy from float vector to Array
  let values = utils::vec_to_array(raw_values);

  // create a window
  let title_str = String::from(title).clone(); // if we don't clone we get a bug
  let wnd = match af::Window::new(window_x as i32, window_y as i32, title_str) {
    Ok(v)  => v,
    Err(e) => panic!("Window creation failed: {}", e), //XXX: handle better
  };

  af::print(&af::min(&values, 0).unwrap());
  af::print(&af::max(&values, 0).unwrap());
  // display till closed
  loop {
    wnd.draw_plot(&af::range(Dim4::new(&[raw_values.len() as u64, 1, 1, 1])
                             , 0, af::Aftype::F32).unwrap(), &values, None);
    if wnd.is_closed().unwrap() == true { break; }
  }
}
