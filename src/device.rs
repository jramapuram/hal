use af;
use af::{Backend};
use std::cell::Cell;
use std::sync::Arc;

pub type DeviceManager = Arc<DeviceManagerFactory>;

#[derive(Clone, Copy, PartialEq)]
pub struct Device {
  pub backend: Backend,
  pub id: i32,
}

pub struct DeviceManagerFactory {
  devices: Vec<Device>,
  current: Cell<Device>,
}

// toggle the backend and device
fn set_device(device: Device) {
  match af::set_backend(device.backend) {
    Ok(_)  => {},
    Err(e) =>  panic!("could not set backend: {:?}", e),
  };

  match af::set_device(device.id) {
    Ok(_)  => {},
    Err(e) =>  panic!("could not set device: {:?}", e),
  };
}

fn create_devices(backend: Backend) -> Vec<Device> {
  let mut buffer: Vec<Device> = Vec::new();
  af::set_backend(backend).unwrap();
  let num_devices: i32 = af::device_count().unwrap();
  for i in 0..num_devices {
    buffer.push(Device{ backend: backend, id: i });
  }
  buffer
}


impl DeviceManagerFactory {
  pub fn new() -> Arc<DeviceManagerFactory> {
    let mut devices = Vec::new();
    let available = af::get_available_backends().unwrap();
    for backend in available {
      devices.extend(create_devices(backend));
    }

    assert!(devices.len() > 0);
    let current = devices.last().unwrap().clone();
    set_device(current);

    Arc::new(DeviceManagerFactory {
      devices: devices,
      current: Cell::new(current),
    })
  }

  pub fn swap(&self, device: Device)
  {
    let c = self.current.get();
    if c != device
    {
      assert!(self.devices.contains(&device));
      println!("Swapping {}/{} to {}/{}", c.backend, c.id
               , device.backend, device.id);
      set_device(device);
      self.current.set(device);
    }
  }
}
