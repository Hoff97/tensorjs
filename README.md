# TensorJS

![Test](https://github.com/Hoff97/tensorjs/workflows/Test/badge.svg?branch=develop)
![Version](https://img.shields.io/npm/v/@hoff97/tensor-js)

This is a JS/TS library for accelerated tensor computation intended to be
run in the browser. It contains an implementation for numpy-style
multidimensional arrays and their operators.

It also allows executing Onnx models. For examples check the [examples folder](https://github.com/Hoff97/tensorjs/tree/develop/examples/).

There are three execution backends available:
- **CPU:** This is implemented in plain javascript and thus
  not very fast. It is intended to be a reference implementation. Big optimizations are avoided for simplicity.
- **Web Assembly:** This is implemented in Rust. It is
  optimized for faster execution (although right now
  there is a lot of work to be done).
- **GPU:** This uses WebGL to enable very fast execution and
  should be used whenever a GPU is available. It is typically
  ~10-100 times faster than the WASM backend (except for
  a few operators). Most of the development focus was spent here
  so this is by far the fastest backend.

# How to use

Install with
```sh
$ npm install @hoff97/tensor-js
```

and then import
```typescript
import * as tjs from '@hoff97/tensor-js';
```
or import the stuff you need directly.

## Tensors

You can create tensors of the respective backend like this:
- CPU:
  ```typescript
  const tensor = new tjs.tensor.cpu.CPUTensor([2,2], [1,2,3,4]);
  ```
- WASM:
  ```typescript
  const tensor = new tjs.tensor.wasm.WASMTensor(new Float32Array([1,2,3,4]), [2,2]);
  ```
- GPU:
  ```typescript
  const tensor = new tjs.tensor.gpu.GPUTensor(new Float32Array([1,2,3,4]), [2,2], 32);
  ```
  or directly from an image/video element:
  ```typescript
  const video: HTMLVideoElement = document.querySelector("#videoElement");
  const tensor = tjs.tensor.gpu.GPUTensor.fromData(video);
  ```
  which will be a tensor with shape `[height,width,4]`.
  Creating a GPU tensor from a video element will usually be pretty fast.
  Creation from an image not necessarily, since here the image data
  first has to be transferred to the GPU.

### Tensor operations

Once you have created a tensor, you can do operations on it, for example:
- Add two tensors
  ```typescript
  const res = a.add(b);
  ```
- Matrix multiplication
  ```typescript
  const res = a.matMul(b);
  ```
- Find the maximum
  ```typescript
  const res = a.max(1);
  ```

For a list of all operators, see the [docs](https://hoff97.github.io/tensorjs/classes/tensor.html).
Most operators will behave like their numpy/pytorch counterparts.

### Reading values

When you want to read data from a tensor:
```typescript
const values = await tensor.getValues();
```
which will give you the values as a array of the values.
For CPU tensors you can also get the value at an index:
```typescript
const value = tensor.get([1,2,3,4]);
```

### Converting between backends

You can conver a tensor to a different backend like so:

```typescript
const cpuTensor = await tjs.util.convert.toCPU(tensor);
const wasmTensor = await tjs.util.convert.toWASM(tensor);
const gpuTensor = await tjs.util.convert.toGPU(tensor);
```

Note that converting to/from a GPU tensor is very expensive and should
be prevented if possible.


## Onnx model support

You can load an onnx model like this:
```typescript
const res = await fetch(`model.onnx`);
const buffer = await res.arrayBuffer();

const model = new tjs.onnx.model.OnnxModel(buffer);
```

You will very likely want to run this model on the GPU. To do this:
```typescript
await model.toGPU();
```

### Optimizations

There are a few optimization passes that can be done on an Onnx model to get faster execution.
To do this, run
```typescript
model.optimize()
```

### Running with half precision

By default full precision floats (32-bits) are used for model execution.
On the GPU backend, you can try executing with
half precision, but be aware that this might not work for all models.
To use half precision, specify this when loading the model:
```typescript
const model = new tjs.onnx.model.OnnxModel(buffer, {
  precision: 16
})
model.toGPU();
```
For the best performance you should also create your GPU tensors with half precision
```typescript
const tensor = new tjs.tensor.gpu.GPUTensor(new Float32Array([1,2,3,4]), [2,2], 16);
```
or directly from an image/video element:
```typescript
const video: HTMLVideoElement = document.querySelector("#videoElement");
const tensor = tjs.tensor.gpu.GPUTensor.fromData(video, 16);
```

The outputs of the model will be half-precision tensors as well.
To read the values of a half precision gpu tensor, you have to convert
it to full precision first, which can be done with:
```typescript
const values = await tensor.copy(32).getValues();
```

### Other performance considerations

Try to run your models with static input sizes. TensorJS will compile specialized versions of all operations
after enough forward passes. For this the input shapes of the tensors have to be constant though.

# Documentation

You can find the documentation [here](https://hoff97.github.io/tensorjs/).

# Development

## Setup

Make sure you have [wasm-pack](https://github.com/rustwasm/wasm-pack) installed, then

```sh
$ npm run build:wasm
$ npm install
```

## Build

```sh
$ npm run build
```

This will first build the rust into a WASM package and
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

## Testing the GPU implementation:

The tests are run in a Headless Chrome by default.
Unfortunately headless chrome relies on a software
implementation for WebGL (instead of Hardware support).
For this reason the GPU backend is not tested by default.

To run GPU tests, uncomment the lines in `./test/gpu.test.ts`,
set `run = true` in `./test/onnx.test.ts`, `./test/onnxPrecompiled.test.ts`
(and optionally `./test/onnxPrecompiled.test.ts` although the model
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