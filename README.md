# hal_rust
Rust based Cross-GPU Machine Learning

## Installation
The following will be for Ubuntu 14.10+.
For other OS's please install all the required [arrayfire dependencies](https://github.com/arrayfire/arrayfire/wiki/Build-Instructions-for-Linux)

Install Rust if you haven't already:
```bash
curl -sSf https://static.rust-lang.org/rustup.sh | sh
```

Install the dependencies:
```bash
sudo apt-get install -y build-essential git subversion cmake libfreeimage-dev libatlas3gf-base libatlas-dev libfftw3-dev liblapacke-dev libboost1.55-all-dev libglew-dev libglewmx-dev libglfw3-dev
```

Clone the repo with the **submodules**:
```bash
git clone --recursive https://github.com/jramapuram/
```

Modify build.conf (located at arrayfire-rust/build.conf) to suit your compute device backend (eg: CUDA or openCL, etc).

Build & run example:
```bash
cargo build
cargo run --example perceptron
```
