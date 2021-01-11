import Tensor from '../../types';

import { compareShapes, getSize } from '../../util/shape';

import { exp } from '../../ops/gpu/exp';
import { log } from '../../ops/gpu/log';
import { sqrt } from '../../ops/gpu/sqrt';
import { add } from '../../ops/gpu/add';
import { subtract } from '../../ops/gpu/subtract';
import { multiply } from '../../ops/gpu/multiply';
import { divide } from '../../ops/gpu/divide';
import { matmul } from '../../ops/gpu/matmul';
import { sum } from '../../ops/gpu/sum';
import { product } from '../../ops/gpu/product';
import { max } from '../../ops/gpu/max';
import { min } from '../../ops/gpu/min';
import { defaultAllocator, gl } from './gl';
import { MemoryEntry } from './memory';
import { conv } from '../../ops/gpu/conv';
import { concat } from '../../ops/gpu/concat';
import { gemm } from '../../ops/gpu/gemm';
import { abs } from '../../ops/gpu/abs';
import { transpose } from '../../ops/gpu/transpose';
import { power } from '../../ops/gpu/power';
import { averagePool } from '../../ops/gpu/averagePool';
import { clip } from '../../ops/gpu/clip';
import { reduceMean } from '../../ops/gpu/reduceMean';
import { repeat } from '../../ops/gpu/repeat';
import { expand } from '../../ops/gpu/expand';
import { copy } from '../../ops/gpu/copy';
import { reduceMeanSquare } from '../../ops/gpu/reduceMeanSquare';
import { sumSquare } from '../../ops/gpu/sumSquare';


export class GPUTensor extends Tensor {
  public memory: MemoryEntry;

  public size: number;

  public shape: readonly number[];

  public deleted: boolean = false;

  constructor(values: Float32Array | MemoryEntry, shape: readonly number[]) {
    super();

    this.size = getSize(shape);
    this.shape = shape;

    if (values instanceof Float32Array) {
      this.memory = defaultAllocator.allocateTexture(values);
    } else {
      this.memory = values;
    }
  }

  getValues(): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      gl({framebuffer: this.memory.frameBuffer})(() => {
        let result = new Float32Array(this.memory.size);
        result = gl.read(result);
        resolve(result.subarray(0, this.size));
      });
    });
  }

  getShape(): readonly number[] {
    return this.shape;
  }

  async gpu(): Promise<GPUTensor> {
    return this;
  }

  delete(): void {
    this.deleted = true;
    defaultAllocator.deallocate(this.memory);
    this.memory = undefined;
  }

  copy(): Tensor {
    return copy(this);
  }

  exp(): Tensor {
    return exp(this);
  }

  log(): Tensor {
    return log(this);
  }

  sqrt(): Tensor {
    return sqrt(this);
  }

  abs(): Tensor {
    return abs(this);
  }

  add_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only add GPU tensor to GPU tensor');
    }
    return add(th, tensor, resultShape);
  }

  subtract_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only subtract GPU tensor from GPU tensor');
    }
    return subtract(th, tensor, resultShape);
  }

  multiply_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only multiply GPU tensor with GPU tensor');
    }
    return multiply(th, tensor, resultShape);
  }

  divide_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only divide GPU tensor by GPU tensor');
    }
    return divide(th, tensor, resultShape);
  }

  power_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only take GPU tensor to power of GPU tensor');
    }
    return power(th, tensor, resultShape);
  }

  matMul(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only matrix multiply GPU tensor to GPU tensor');
    }
    return matmul(this, tensor);
  }

  gemm_impl(b: Tensor, aTranspose: boolean, bTranspose: boolean, alpha: number, beta: number, c?: Tensor): Tensor {
    if (!(b instanceof GPUTensor && (c === undefined || c instanceof GPUTensor))) {
      throw new Error('Can only do gemm with CPU tensors');
    }
    return gemm(this, b, aTranspose, bTranspose, alpha, beta, c as GPUTensor);
  }

  sum_impl(axes: number[], keepDims: boolean): Tensor {
    return sum(this, axes, keepDims);
  }

  sumSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return sumSquare(this, axes, keepDims);
  }

  reduceMean_impl(axes: number[], keepDims: boolean): Tensor {
    return reduceMean(this, axes, keepDims);
  }

  reduceMeanSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return reduceMeanSquare(this, axes, keepDims);
  }

  product_impl(axes: number[], keepDims: boolean): Tensor {
    return product(this, axes, keepDims);
  }

  max_impl(axes: number[], keepDims: boolean): Tensor {
    return max(this, axes, keepDims);
  }

  min_impl(axes: number[], keepDims: boolean): Tensor {
    return min(this, axes, keepDims);
  }

  conv_impl(kernel: Tensor, dilations: number[], group: number, pads: number[], strides: number[], bias?: Tensor): Tensor {
    if (!(kernel instanceof GPUTensor) || (bias !== undefined && !(bias instanceof GPUTensor))) {
      throw new Error('Can only do convolution of GPU tensor with GPU tensor');
    }
    return conv(this, kernel, dilations, group, pads, strides, bias as GPUTensor);
  }

  averagePool_impl(kernelShape: number[], pads: number[], strides: number[], includePad: boolean): Tensor {
    return averagePool(this, kernelShape, pads, strides, includePad);
  }

  reshape_impl(shape: number[]): Tensor {
    return copy(this, shape);
  }

  concat(tensor: Tensor, axis: number): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only concat GPU tensor to GPU tensor');
    }
    return concat(this, tensor, axis);
  }

  transpose_impl(permutation: number[]): Tensor {
    return transpose(this, permutation);
  }

  clip(min?: number, max?: number): Tensor {
    return clip(this, min, max);
  }

  repeat(repeats: number[]): Tensor {
    return repeat(this, repeats);
  }

  expand(shape: number[]): Tensor {
    const [_shape, goal, resultShape] = this.alignShapes(this.shape, shape);
    if (compareShapes(this.shape, resultShape)) {
      return this.copy();
    }
    return expand(this.reshape(_shape) as GPUTensor, resultShape);
  }
}
