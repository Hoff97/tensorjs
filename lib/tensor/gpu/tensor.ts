import Tensor, { PadMode } from '../../types';

import { compareShapes, getSize } from '../../util/shape';

import { log } from '../../ops/gpu/log';
import { sqrt } from '../../ops/gpu/sqrt';
import { MatMulOperation } from '../../ops/gpu/matmul';
import { sum } from '../../ops/gpu/sum';
import { product } from '../../ops/gpu/product';
import { max } from '../../ops/gpu/max';
import { min } from '../../ops/gpu/min';
import { defaultAllocator, gl } from './gl';
import { MemoryEntry } from './memory';
import { concat } from '../../ops/gpu/concat';
import { gemm } from '../../ops/gpu/gemm';
import { transpose } from '../../ops/gpu/transpose';
import { power } from '../../ops/gpu/power';
import { clip } from '../../ops/gpu/clip';
import { reduceMean } from '../../ops/gpu/reduceMean';
import { repeat } from '../../ops/gpu/repeat';
import { expand } from '../../ops/gpu/expand';
import { copy } from '../../ops/gpu/copy';
import { reduceMeanSquare } from '../../ops/gpu/reduceMeanSquare';
import { sumSquare } from '../../ops/gpu/sumSquare';
import { padOp } from '../../ops/gpu/pad';
import { CPUTensor } from '../cpu/tensor';
import { gather } from '../../ops/gpu/gather';
import { floor } from '../../ops/gpu/floor';
import { ceil } from '../../ops/gpu/ceil';
import { slice } from '../../ops/gpu/slice';
import { upsample } from '../../ops/gpu/upsample';
import REGL from 'regl';
import { toTexture } from '../../ops/gpu/toTexture';
import { normalize } from '../../ops/gpu/normalize';
import { ExpOperation } from '../../ops/gpu/exp';
import { GPUTensorConstructor, GPUTensorI } from './interface';
import { ConvBiasOperation, ConvOperation } from '../../ops/gpu/conv';
import { AbsOperation } from '../../ops/gpu/abs';
import { AddOperation } from '../../ops/gpu/add';
import { MultiplyOperation } from '../../ops/gpu/multiply';
import { SubtractOperation } from '../../ops/gpu/subtract';
import { DivideOperation } from '../../ops/gpu/divide';
import { AveragePoolOperation } from '../../ops/gpu/averagePool';


export class GPUTensor extends Tensor implements GPUTensorI {
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

  static fromData(data: REGL.TextureImageData) {
    const texture = gl.texture({
      data: data,
      format: "rgba",
      type: "float"
    });

    const memory = defaultAllocator.allocateFramebuffer(texture);

    const width = texture.width;
    const height = texture.height;

    return new GPUTensor(memory, [height, width, 4]);
  }

  toTexture(): GPUTensor {
    return toTexture(this);
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
    return defaultExp.calc({input: this}) as any;
  }

  log(): Tensor {
    return log(this);
  }

  sqrt(): Tensor {
    return sqrt(this);
  }

  abs(): Tensor {
    return defaultAbs.calc({input: this});
  }

  floor(): Tensor {
    return floor(this);
  }

  ceil(): Tensor {
    return ceil(this);
  }

  add_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only add GPU tensor to GPU tensor');
    }
    return defaultAdd.calc({A: th, B: tensor, outputShape: resultShape});
  }

  subtract_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only subtract GPU tensor from GPU tensor');
    }
    return defaultSubtract.calc({A: th, B: tensor, outputShape: resultShape});
  }

  multiply_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only multiply GPU tensor with GPU tensor');
    }
    return defaultMultiply.calc({A: th, B: tensor, outputShape: resultShape});
  }

  divide_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only divide GPU tensor by GPU tensor');
    }
    return defaultDivide.calc({A: th, B: tensor, outputShape: resultShape});
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
    return defaultMatMul.calc({A: this, B: tensor});
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

    if (bias === undefined) {
      return defaultConv.calc({
        X: this,
        W: kernel,
        pads, dilations, strides
      });
    } else {
      return defaultConvBias.calc({
        X: this,
        W: kernel,
        B: bias as GPUTensor,
        pads, dilations, strides
      });
    }
  }

  averagePool_impl(kernelShape: number[], pads: number[], strides: number[], includePad: boolean): Tensor {
    return defaultAveragePool.calc({
      X: this,
      includePad,
      kernelShape,
      pads,
      strides
    });
  }

  reshape_impl(shape: number[], _copy: boolean): Tensor {
    if (_copy) {
      return copy(this, shape);
    } else {
      return new GPUTensor(this.memory, shape);
    }
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
    return expand(this.reshape(_shape, false) as GPUTensor, resultShape);
  }

  pad_impl(pads: number[], mode: PadMode, value: number): Tensor {
    return padOp(this, pads, mode, value);
  }

  gather(axis: number, indices: CPUTensor): Tensor {
    return gather(this, axis, indices);
  }

  slice_impl(starts: number[], ends: number[], axes: number[]): Tensor {
    return slice(this, starts, ends, axes);
  }

  upsample(scales: number[]): Tensor {
    return upsample(this, scales);
  }

  normalize(mean: Tensor, variance: Tensor, epsilon: number, scale: Tensor, bias: Tensor): Tensor {
    if (!(mean instanceof GPUTensor) || !(variance instanceof GPUTensor) || !(scale instanceof GPUTensor) || !(bias instanceof GPUTensor)) {
      throw new Error('Can only normalize with CPU tensors');
    }
    return normalize(this, mean, variance, epsilon, scale, bias);
  }
}

const constructor: GPUTensorConstructor<GPUTensor> = (a: MemoryEntry,b: readonly number[]) => new GPUTensor(a,b);

const defaultMatMul = new MatMulOperation(constructor);
const defaultExp = new ExpOperation(constructor);
const defaultConv = new ConvOperation(constructor);
const defaultAveragePool = new AveragePoolOperation(constructor);
const defaultConvBias = new ConvBiasOperation(constructor);
const defaultAbs = new AbsOperation(constructor);
const defaultAdd = new AddOperation(constructor);
const defaultSubtract = new SubtractOperation(constructor);
const defaultMultiply = new MultiplyOperation(constructor);
const defaultDivide = new DivideOperation(constructor);