# Development

This file briefly explains how to set up a development environment.

## Structure

This project contains the following directories
- `lib`: The main library implementation in typescript
- `src`: The implementation of the WASM backend in Rust. The build output of this is
    copied into the `lib` directory.
  - `src/test`: Contains rust tests for the WASM backend, mainly used for debugging
- `test`: All tests for this library
  - `test/performance`: Performance tests
- `tools`: Other tools needed for development
- `examples`: Several example applications using tensorjs

## Setup

Make sure you have [wasm-pack](https://github.com/rustwasm/wasm-pack) installed, then

```sh
$ npm install
```

## Build

```sh
$ npm run build
```

This will first build the rust code into a WASM package and
then build the whole library.

## Testing

Before the first run, data for the onnx operator tests has to be loaded:

```sh
$ npm run testData
```

then

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

Coverage reports are generated in `coverage`.

### Debugging tests

There are 2 launch configurations in `.vscode/launch.json`:
- "Debug unit tests in library 'rust-wasm-tensor'": These debug the rust tests defined in `src/tests/` and
  are useful for debugging the WASM backend.
- "Debug tests in firefox": This allows you to debug the typescript tests. This uses Firefox so
  you need Firefox installed on your computer and the "Debugger for Firefox" extension.

### Testing the GPU implementation:

The tests are run in a Headless Chrome by default.
Unfortunately headless chrome relies on a software
implementation for WebGL (instead of Hardware support).
For this reason the GPU backend is not tested by default.

To run GPU tests, uncomment the lines in `./test/gpu.test.ts`,
set `run = true` in `./test/onnx.test.ts`, `./test/onnxPrecompiled.test.ts`
(and optionally `./test/onnxModel.test.ts` although the model
tests are quite performance intensive and should only be run on a suitable
computer),
change the `browsers` field in `karma.conf.js` to `ChromeGPU`.
If Chrome still doesnt use the GPU, you can try something
like

`export DISPLAY=:0`

or

`Xvfb -ac :0 -screen 0 1280x1024x16 & export DISPLAY=:0`

on Linux.

## Performance tests

Performance tests can be run with `npm run test:performance`.
As in the unit tests, Chrome doesnt use the GPU by default.
If you want to run performance tests with the GPU,
change the `browsers` field in `karma.performance.js` to `ChromeGPU` and
optionally follow the instructions of the paragraph above.

Additionally, the rust implementation can be benchmarked by running
```
$ npm run test:rustBenchmark
```
You can also run individual benchmarks with
```
$ cargo +nightly bench --features benchmark BENCHMARK_NAME
```

All rust benchmarks are contained in `src/test/tensor/benchmark.rs`.