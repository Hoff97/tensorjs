# Contributing to tensor-js

There are several ways to contribute to tensor-js:

- You can propose new features. For this open an [issue](https://github.com/Hoff97/tensorjs/issues/new)
- You can request new onnx operators. Like above you can open an [issue](https://github.com/Hoff97/tensorjs/issues/new)
- You can help implementing new features. For this, see the paragraphs below
- You can implement new examples. While there are some in the [examples folder](https://github.com/Hoff97/tensorjs/tree/master/examples), there could definitely be more!

# Implementing new features

Before starting with a feature, please check the github [issues](https://github.com/Hoff97/tensorjs/issues) and see
if a corresponding issue exists.

For a description of the development setup, check [Development.md](https://github.com/Hoff97/tensorjs/blob/develop/doc/Development.md).

You should generally implement new features by forking the repository and opening a Pull request against the **develop** branch. Pull requests will
only be merged if all tests pass and no bad linting errors are added. If you are using VSCode, files should be automatically formatted on
save, which will prevent many linting errors.

## Implementing a new operator

If you want to implement a new operator, the process is typically the following:

- Add the function declaration to the `Tensor` class in `lib/types.ts`
- Implement the operator on some (or in the best case all) backends:
  - CPU backend:
    - Its probably wise to start with this backend, since no special knowledge is required for this.
    - Implement the operator in a single file in `lib/ops/cpu`
    - Bind the implementation in `lib/tensor/cpu/tensor.ts`. This should ideally be only very few lines of code.
  - WASM backend:
    - Implement the operation in `src/tensor.rs`.
    - Then build the wasm backend with `npm run build:wasm`
    - Bind the implementation in `lib/tensor/wasm/tensor.ts`. This should ideally be only very few lines of code.
  - GPU backend:
    - Implement an operation in `lib/ops/gpu`. Check `lib/ops/gpu/operation.ts` and other GPU implementations to understand how this works
    - Bind the implementation in `lib/tensor/gpu/tensor.ts`.
      - For this, create a new operation dispatcher at the bottom (where all the dispatchers live)
      - Call the `calc` method in the GPUTensor implementation
- If a matching onnx operator exists, you can add a layer definition in `lib/onnx/nodes` and add it to `lib/onnx/resolve.ts`
  - In this case, enable the correct test case in `test/data/enabledTests.ts`
- If you want to, you can implement the respective backward operation in `lib/autograd/variable.ts` for autograd support.