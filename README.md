# TensorJS

This is a JS/TS library for accelerated tensor computation intended to be
run in the browser. It contains an implementation for numpy-style
multidimensional arrays and their operators.

There are three execution backends available:
- **CPU:** This is implemented in plain javascript and thus
  not very fast. It is intended to be a reference implementation. Big optimizations are avoided for simplicity.
- **Web Assembly:** This is implemented in Rust. It is
  optimized for faster execution (although right now
  there is a lot of work to be done).
- **GPU:** This uses WebGL to enable very fast execution and
  should be used whenever a GPU is available. It is typically
  ~10-100 times faster than the WASM backend (except for
  a few operators).


## Development

### Setup

Make sure you have [wasm-pack](https://github.com/rustwasm/wasm-pack) installed, then

```sh
$ npm run build:wasm
$ npm install
```

### Build

```sh
$ npm run build
```

This will first build the rust into a WASM package and
then build the whole library.

### Testing

```sh
$ npm run test
```

This will run:
- Rust tests. These are run as a standard binary (no
  Web assembly) and can be debugged, by running
  the launch configuration named *Debug unit tests in library 'rust-wasm-tensor'* in VSCode.
  They can also be run individually: `npm run test:rust`
- Unit tests for all backends, which can be run individually
  by `npm run test:integration`

### Testing the GPU implementation:

The tests are run in a Headless Chrome by default.
Unfortunately headless chrome relies on a software
implementation for WebGL (instead of Hardware support).
For this reason the GPU backend is not tested by default.

To run GPU tests, uncomment the lines in `./test/gpu.test.ts` and
change the `browsers` field in `karma.conf.js` to `ChromeGPU`.
If Chrome still doesnt use the GPU, you can try something
like

`export DISPLAY=:0`

or

`Xvfb -ac :0 -screen 0 1280x1024x16 & export DISPLAY=:0`

on Linux.

### Performance tests

Performance tests can be run with `npm run test:performance`.
As in the unit tests, Chrome doesnt use the GPU by default.
If you want to run performance tests with the GPU,
change the `browsers` field in `karma.performance.js` to `ChromeGPU` and
optionally follow the instructions of the paragraph above.