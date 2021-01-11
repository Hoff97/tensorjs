import Tensor, { PadMode } from '../../types';
import { compareShapes } from '../../util/shape';

import { Tensor as WT } from '../../wasm/rust_wasm_tensor';

let WASMT: typeof WT;
export let wasmLoaded: Promise<void> = new Promise((resolve, reject) => {
  import('../../wasm/rust_wasm_tensor').then(x => {
    WASMT = x.Tensor;
    resolve();
  });
});

export class WASMTensor extends Tensor {
  public wasmTensor: WT;

  constructor(values: Float32Array | WT, shape?: Uint32Array) {
    super();

    if (values instanceof Float32Array) {
      if (shape === undefined) {
        throw new Error('Need the shape when creating a Wasm tensor from values');
      }
      this.wasmTensor = WASMT.create(shape, values);
    } else {
      this.wasmTensor = values;
    }
  }

  getValues() {
    return Promise.resolve(this.wasmTensor.get_vals());
  }

  getShape(): readonly number[] {
    return Array.from(this.wasmTensor.get_shape());
  }

  async wasm(): Promise<WASMTensor> {
    return this;
  }

  delete(): void {
    this.wasmTensor.free();
    this.wasmTensor = undefined;
  }

  copy(): Tensor {
    return new WASMTensor(this.wasmTensor.copy());
  }

  exp(): Tensor {
    return new WASMTensor(this.wasmTensor.exp());
  }

  log(): Tensor {
    return new WASMTensor(this.wasmTensor.log());
  }

  sqrt(): Tensor {
    return new WASMTensor(this.wasmTensor.sqrt());
  }

  abs(): Tensor {
    return new WASMTensor(this.wasmTensor.abs());
  }

  add_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only add WASM tensor to WASM tensor');
    }

    return new WASMTensor(th.wasmTensor.addition(tensor.wasmTensor));
  }

  subtract_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only subtract WASM tensor from WASM tensor');
    }
    return new WASMTensor(th.wasmTensor.subtraction(tensor.wasmTensor));
  }

  multiply_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only multiply WASM tensor with WASM tensor');
    }
    return new WASMTensor(th.wasmTensor.multiply(tensor.wasmTensor));
  }

  divide_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only divide WASM tensor by WASM tensor');
    }
    return new WASMTensor(th.wasmTensor.divide(tensor.wasmTensor));
  }

  power_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only take WASM tensor to power of WASM tensor');
    }
    return new WASMTensor(th.wasmTensor.power(tensor.wasmTensor));
  }

  matMul(tensor: Tensor): Tensor {
    if (!(tensor instanceof WASMTensor)) {
      throw new Error('Can only add WASM tensor to WASM tensor');
    }
    return new WASMTensor(this.wasmTensor.matmul(tensor.wasmTensor));
  }

  gemm_impl(b: Tensor, aTranspose: boolean, bTranspose: boolean, alpha: number, beta: number, c?: Tensor): Tensor {
    if (!(b instanceof WASMTensor && (c === undefined || c instanceof WASMTensor))) {
      throw new Error('Can only do gemm with CPU tensors');
    }
    if (c !== undefined) {
      return new WASMTensor(this.wasmTensor.gemm_with_c(b.wasmTensor, aTranspose, bTranspose, alpha, (c as WASMTensor).wasmTensor, beta));
    } else {
      return new WASMTensor(this.wasmTensor.gemm(b.wasmTensor, aTranspose, bTranspose, alpha));
    }
  }

  sum_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(this.wasmTensor.sum(new Uint32Array(axes), keepDims));
  }

  sumSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(this.wasmTensor.sum_square(new Uint32Array(axes), keepDims));
  }

  product_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(this.wasmTensor.product(new Uint32Array(axes), keepDims));
  }

  max_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(this.wasmTensor.max(new Uint32Array(axes), keepDims));
  }

  min_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(this.wasmTensor.min(new Uint32Array(axes), keepDims));
  }

  reduceMean_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(this.wasmTensor.reduce_mean(new Uint32Array(axes), keepDims));
  }

  reduceMeanSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(this.wasmTensor.reduce_mean_square(new Uint32Array(axes), keepDims));
  }

  conv_impl(kernel: Tensor, dilations: number[], group: number, pads: number[], strides: number[], bias?: Tensor): Tensor {
    if (!(kernel instanceof WASMTensor) || (bias !== undefined && !(bias instanceof WASMTensor))) {
      throw new Error('Can only do convolution of CPU tensor with CPU tensor');
    }
    if (bias !== undefined) {
      return new WASMTensor(this.wasmTensor.conv_with_bias(kernel.wasmTensor, (bias as WASMTensor).wasmTensor, new Uint32Array(dilations), group, new Uint32Array(pads), new Uint32Array(strides)));
    } else {
      return new WASMTensor(this.wasmTensor.conv(kernel.wasmTensor, new Uint32Array(dilations), group, new Uint32Array(pads), new Uint32Array(strides)));
    }
  }

  averagePool_impl(kernelShape: number[], pads: number[], strides: number[], includePad: boolean): Tensor {
    return new WASMTensor(this.wasmTensor.average_pool(new Uint32Array(kernelShape), new Uint32Array(pads), new Uint32Array(strides), includePad));
  }

  reshape_impl(shape: number[]): Tensor {
    const sh = new Uint32Array(shape);
    return new WASMTensor(this.wasmTensor.reshape(sh), sh);
  }

  concat(tensor: Tensor, axis: number): Tensor {
    if (!(tensor instanceof WASMTensor)) {
      throw new Error('Can only concat WASM tensor to WASM tensor');
    }
    return new WASMTensor(this.wasmTensor.concat(tensor.wasmTensor, axis));
  }

  transpose_impl(permutation: number[]): Tensor {
    return new WASMTensor(this.wasmTensor.transpose(new Uint32Array(permutation)));
  }

  clip(min?: number, max?: number): Tensor {
    if (min !== undefined && max !== undefined) {
      return new WASMTensor(this.wasmTensor.clip(min, max));
    } else if (min !== undefined ) {
      return new WASMTensor(this.wasmTensor.clip_min(min));
    } else if (max !== undefined) {
      return new WASMTensor(this.wasmTensor.clip_max(max));
    }
    return this.copy();
  }

  repeat(repeats: number[]): Tensor {
    return new WASMTensor(this.wasmTensor.repeat(new Uint32Array(repeats)));
  }

  expand(shape: number[]): Tensor {
    const thisShape = this.getShape();

    const [_shape, goal, resultShape] = this.alignShapes(thisShape, shape);
    if (compareShapes(thisShape, resultShape)) {
      return this.copy();
    }

    const reshaped = this.reshape(_shape) as WASMTensor;

    return new WASMTensor(reshaped.wasmTensor.expand(new Uint32Array(resultShape)));
  }

  static padModeToInt = {
    'constant': 0,
    'reflect': 1,
    'edge': 2
  }

  pad_impl(pads: number[], mode: PadMode, value: number): Tensor {
    return new WASMTensor(this.wasmTensor.pad(new Uint32Array(pads), WASMTensor.padModeToInt[mode],value))
  }
}
