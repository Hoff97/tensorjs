# TensorJS

![Test](https://github.com/Hoff97/tensorjs/workflows/Test/badge.svg?branch=develop)
![Version](https://img.shields.io/npm/v/@hoff97/tensor-js)
![David](https://img.shields.io/david/Hoff97/tensorjs)

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
```shell
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

To see all supported operators, check the [supported operator list](https://github.com/Hoff97/tensorjs/blob/master/Operators.md).

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

## Autograd functionality

Automatic differentiation is supported. For this create variables from all your tensors:

```typescript
const a = new tjs.tensor.cpu.CPUTensor([2,2], [1,2,3,4]);
const b = new tjs.tensor.cpu.CPUTensor([2,2], [5,6,7,8]);

const varA = new tjs.autograd.Variable(a);
const varB = new tjs.autograd.Variable(b);
```

Or use the utility methods:
```typescript
const varA = tjs.autograd.Variable.create([2,2], [1,2,3,4], 'GPU');
const videoElement = document.querySelector("#videoElement");
const varB = tjs.autograd.Variable.fromData(videoElement);
```

Afterwards you can perform normal tensor operations:

```typescript
const mul = varA.matMul(varB);
const sum = mul.sum();
```

To perform a backward pass, call backward on a scalar tensor (a tensor with shape `[1]`).
All variables will have an attribute `.grad`, which is the gradient
```typescript
sum.backward();

console.log(varA.grad);
```

Multiple backward passes will add up the gradients.
After you are done with the variable, delete the computation graph by calling `delete()`.

# Documentation

You can find the documentation [here](https://hoff97.github.io/tensorjs/).

# Development

See [Development.md](https://github.com/Hoff97/tensorjs/blob/develop/Development.md)