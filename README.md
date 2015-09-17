# HAL : Hyper Adaptive Learning
Rust based Cross-GPU Machine Learning

## Installation
The following will be for Ubuntu 14.10+.
For other OS's please install all the required dependencies[[linux](https://github.com/arrayfire/arrayfire/wiki/Build-Instructions-for-Linux) /[osx](https://github.com/arrayfire/arrayfire/wiki/Build-Instructions-for-OSX)]

  1. Install Rust if you haven't already [currently only tested on rustc 1.3+] :
   
  ```bash
  $ curl -sSf https://static.rust-lang.org/rustup.sh | sh
  ```
  2. Install the dependencies:
  
  ```bash
  $ sudo apt-get install -y build-essential git subversion cmake libfreeimage-dev libatlas3gf-base libatlas-dev libfftw3-dev liblapacke-dev libboost1.55-all-dev libglew-dev libglewmx-dev libglfw3-dev
  ```
  3. Clone the repo with the **submodules**:

  ```bash
  $ git clone --recursive https://github.com/jramapuram/hal_rust
  ```
  4. Modify `build.conf` (located at `arrayfire-rust/build.conf`) to suit your compute device backend (eg: CUDA or openCL, etc).
  5. Build & run example:
  
  ```bash
  $ cargo build
  $ cargo run --example perceptron
  ```
#### Example CUDA Ubuntu 14.10 `build.conf`
```json
{
    "use_backend": "cuda",

    "use_lib": false,
    "lib_dir": "/usr/local/lib",
    "inc_dir": "/usr/local/include",

    "build_type": "Release",
    "build_threads": "4",
    "build_cuda": "ON",
    "build_opencl": "OFF",
    "build_cpu": "ON",
    "build_examples": "OFF",
    "build_test": "OFF",
    "build_graphics": "ON",

    "glew_static": "OFF",
    "freeimage_type": "DYNAMIC",
    "cpu_fft_type": "FFTW",
    "cpu_blas_type": "LAPACKE",
    "cpu_lapack_type": "LAPACKE",

    "freeimage_dir": "",
    "fftw_dir": "",
    "acml_dir": "",
    "mkl_dir": "",
    "lapacke_dir": "",
    "glew_dir": "",
    "glfw_dir": "",
    "boost_dir": "",

    "cuda_sdk": "/usr/local/cuda",
    "opencl_sdk": "/usr",
    "sdk_lib_dir": "lib64"
}
```
#### Example CUDA OS X  `build.conf`
```json
{
    "use_backend": "cuda",

    "use_lib": false,
    "lib_dir": "/usr/local/lib",
    "inc_dir": "/usr/local/include",

    "build_type": "Release",
    "build_threads": "4",
    "build_cuda": "ON",
    "build_opencl": "OFF",
    "build_cpu": "OFF",
    "build_examples": "OFF",
    "build_test": "OFF",
    "build_graphics": "ON",

    "glew_static": "OFF",
    "freeimage_type": "DYNAMIC",
    "cpu_fft_type": "FFTW",
    "cpu_blas_type": "LAPACKE",
    "cpu_lapack_type": "LAPACKE",

    "freeimage_dir": "",
    "fftw_dir": "",
    "acml_dir": "",
    "mkl_dir": "",
    "lapacke_dir": "",
    "glew_dir": "",
    "glfw_dir": "",
    "boost_dir": "",

    "cuda_sdk": "/usr/local/cuda",
    "opencl_sdk": "/usr",
    "sdk_lib_dir": "lib"
}
```

#### Example OpenCL OSX `build.conf`
```json
{
    "use_backend": "opencl",

    "use_lib": false,
    "lib_dir": "/usr/local/lib",
    "inc_dir": "/usr/local/include",

    "build_type": "Release",
    "build_threads": "4",
    "build_cuda": "OFF",
    "build_opencl": "ON",
    "build_cpu": "OFF",
    "build_examples": "OFF",
    "build_test": "OFF",
    "build_graphics": "ON",

    "glew_static": "OFF",
    "freeimage_type": "DYNAMIC",
    "cpu_fft_type": "FFTW",
    "cpu_blas_type": "LAPACKE",
    "cpu_lapack_type": "LAPACKE",

    "freeimage_dir": "",
    "fftw_dir": "",
    "acml_dir": "",
    "mkl_dir": "",
    "lapacke_dir": "",
    "glew_dir": "",
    "glfw_dir": "",
    "boost_dir": "",

    "cuda_sdk": "/usr/local/cuda",
    "opencl_sdk": "/usr",
    "sdk_lib_dir": "lib"
}
```
