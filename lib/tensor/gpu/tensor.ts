import Tensor, { Activation, PadMode, Precision } from '../../types';

import { compareShapes, getSize } from '../../util/shape';

import { MatMulOperation } from '../../ops/gpu/matMul/matmul';
import { defaultAllocator, gl } from './gl';
import { GPUMemoryAllocator, MemoryEntry } from './memory';
import { CPUTensor } from '../cpu/tensor';
import REGL from 'regl';
import { ExpOperation } from '../../ops/gpu/unary/exp';
import { GPUTensorConstructor, GPUTensorI } from './interface';
import { ConvBiasOperation, ConvOperation } from '../../ops/gpu/conv/conv';
import { AbsOperation } from '../../ops/gpu/unary/abs';
import { AddOperation } from '../../ops/gpu/binary/add';
import { MultiplyOperation } from '../../ops/gpu/binary/multiply';
import { SubtractOperation } from '../../ops/gpu/binary/subtract';
import { DivideOperation } from '../../ops/gpu/binary/divide';
import { AveragePoolOperation } from '../../ops/gpu/conv/averagePool';
import { ReduceMeanOperation } from '../../ops/gpu/pool/reduceMean';
import { ReduceMeanSquareOperation } from '../../ops/gpu/pool/reduceMeanSquare';
import { SumSquareOperation } from '../../ops/gpu/pool/sumSquare';
import { SumOperation } from '../../ops/gpu/pool/sum';
import { ProductOperation } from '../../ops/gpu/pool/product';
import { MaxOperation } from '../../ops/gpu/pool/max';
import { MinOperation } from '../../ops/gpu/pool/min';
import { CeilOperation } from '../../ops/gpu/unary/ceil';
import { ClipOperation } from '../../ops/gpu/unary/clip';
import { FloorOperation } from '../../ops/gpu/unary/floor';
import { ConcatOperation } from '../../ops/gpu/util/concat';
import { CopyOperation } from '../../ops/gpu/util/copy';
import { ExpandOperation } from '../../ops/gpu/util/expand';
import { GatherOperation } from '../../ops/gpu/util/gather';
import { GemmCOperation, GemmOperation } from '../../ops/gpu/matMul/gemm';
import { PowerOperation } from '../../ops/gpu/binary/power';
import { SqrtOperation } from '../../ops/gpu/unary/sqrt';
import { LogOperation } from '../../ops/gpu/unary/log';
import { TransposeOperation } from '../../ops/gpu/util/transpose';
import { RepeatOperation } from '../../ops/gpu/util/repeat';
import { PadOperation } from '../../ops/gpu/conv/pad';
import { SliceOperation } from '../../ops/gpu/util/slice';
import { UpsampleOperation } from '../../ops/gpu/conv/upsample';
import { NormalizeOperation } from '../../ops/gpu/conv/normalize';


export class GPUTensor extends Tensor implements GPUTensorI {
  public memory: MemoryEntry;

  public size: number;

  public deleted: boolean = false;

  constructor(values: Float32Array | MemoryEntry, public shape: readonly number[], public precision: Precision) {
    super();

    this.size = getSize(shape);

    if (values instanceof Float32Array) {
      this.memory = defaultAllocator.allocateTexture(values, precision);
    } else {
      this.memory = values;
    }
  }

  static fromData(data: REGL.TextureImageData, precision: Precision) {
    const texture = gl.texture({
      data: data,
      format: "rgba",
      type: precision === 32 ? "float" : "half float"
    });

    const memory = defaultAllocator.allocateFramebuffer(texture, precision);

    const width = texture.width;
    const height = texture.height;

    return new GPUTensor(memory, [height, width, 4], precision);
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

  delete(allocator?: GPUMemoryAllocator): void {
    this.deleted = true;
    if (allocator !== undefined) {
      allocator.deallocate(this.memory);
    } else {
      defaultAllocator.deallocate(this.memory);
    }
    this.memory = undefined;
  }

  copy(): Tensor {
    return defaultCopy.calc({input: this});
  }

  exp(): Tensor {
    return defaultExp.calc({input: this}) as any;
  }

  log(): Tensor {
    return defaultLog.calc({input: this}) as any;
  }

  sqrt(): Tensor {
    return defaultSqrt.calc({input: this}) as any;
  }

  abs(): Tensor {
    return defaultAbs.calc({input: this});
  }

  floor(): Tensor {
    return defaultFloor.calc({input: this});
  }

  ceil(): Tensor {
    return defaultCeil.calc({input: this})
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
    return defaultPower.calc({A: th, B: tensor, outputShape: resultShape});
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
    if (c === undefined) {
      return defaultGemm.calc({a: this, b, aTranspose, bTranspose, alpha, beta});
    } else {
      return defaultGemmC.calc({a: this, b, c: c as GPUTensor, aTranspose, bTranspose, alpha, beta});
    }
  }

  sum_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultSum.calc({X: this, axes, keepDims});
  }

  sumSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultSumSquare.calc({X: this, axes, keepDims});
  }

  reduceMean_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultMean.calc({X: this, axes, keepDims});
  }

  reduceMeanSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultMeanSquare.calc({X: this, axes, keepDims});
  }

  product_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultProduct.calc({X: this, axes, keepDims});
  }

  max_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultMax.calc({X: this, axes, keepDims});
  }

  min_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultMin.calc({X: this, axes, keepDims});
  }

  conv_impl(kernel: Tensor, dilations: number[], group: number, pads: number[], strides: number[], bias?: Tensor, activation?: Activation): Tensor {
    if (!(kernel instanceof GPUTensor) || (bias !== undefined && !(bias instanceof GPUTensor))) {
      throw new Error('Can only do convolution of GPU tensor with GPU tensor');
    }

    if (bias === undefined) {
      return defaultConv.calc({
        X: this,
        W: kernel,
        pads, dilations, strides,
        activation
      });
    } else {
      return defaultConvBias.calc({
        X: this,
        W: kernel,
        B: bias as GPUTensor,
        pads, dilations, strides,
        activation
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
      return defaultCopy.calc({input: this, outputShape: shape});
    } else {
      return new GPUTensor(this.memory, shape, this.precision);
    }
  }

  concat(tensor: Tensor, axis: number): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only concat GPU tensor to GPU tensor');
    }
    return defaultConcat.calc({A: this, B: tensor, axis});
  }

  transpose_impl(permutation: number[]): Tensor {
    return defaultTranspose.calc({A: this, permutation});
  }

  clip(min?: number, max?: number): Tensor {
    return defaultClip.calc({input: this, minVal: min, maxVal: max});
  }

  repeat(repeats: number[]): Tensor {
    return defaultRepeat.calc({A: this, repeats});
  }

  expand(shape: number[]): Tensor {
    const [_shape, goal, resultShape] = this.alignShapes(this.shape, shape);
    if (compareShapes(this.shape, resultShape)) {
      return this.copy();
    }
    return defaultExpand.calc({input: this.reshape(_shape, false) as GPUTensor, outputShape: resultShape});
  }

  pad_impl(pads: number[], mode: PadMode, value: number): Tensor {
    return defaultPad.calc({input: this, pads, mode, value});
  }

  gather(axis: number, indices: CPUTensor): Tensor {
    return defaultGather.calc({X: this, axis, indices});
  }

  slice_impl(starts: number[], ends: number[], axes: number[]): Tensor {
    return defaultSlice.calc({X: this, starts, ends, axes});
  }

  upsample(scales: number[]): Tensor {
    return defaultUpsample.calc({X: this, scales});
  }

  normalize(mean: Tensor, variance: Tensor, epsilon: number, scale: Tensor, bias: Tensor): Tensor {
    if (!(mean instanceof GPUTensor) || !(variance instanceof GPUTensor) || !(scale instanceof GPUTensor) || !(bias instanceof GPUTensor)) {
      throw new Error('Can only normalize with CPU tensors');
    }

    return defaultNormalize.calc({
      X: this,
      Mean: mean,
      Variance: variance,
      Scale: scale,
      Bias: bias,
      epsilon
    });
  }
}

export const gpuConstructor: GPUTensorConstructor<GPUTensor> = (a: MemoryEntry,b: readonly number[], precision: Precision) => new GPUTensor(a,b,precision);

const defaultMatMul = new MatMulOperation(gpuConstructor);
const defaultGemm = new GemmOperation(gpuConstructor);
const defaultGemmC = new GemmCOperation(gpuConstructor);

//Unary operations
const defaultExp = new ExpOperation(gpuConstructor);
const defaultAbs = new AbsOperation(gpuConstructor);
const defaultCeil = new CeilOperation(gpuConstructor);
const defaultFloor = new FloorOperation(gpuConstructor);
const defaultClip = new ClipOperation(gpuConstructor);
const defaultSqrt = new SqrtOperation(gpuConstructor);
const defaultLog = new LogOperation(gpuConstructor);

//Convolutions
const defaultConv = new ConvOperation(gpuConstructor);
const defaultAveragePool = new AveragePoolOperation(gpuConstructor);
const defaultConvBias = new ConvBiasOperation(gpuConstructor);
const defaultPad = new PadOperation(gpuConstructor);
const defaultUpsample = new UpsampleOperation(gpuConstructor);

//Binary operations
const defaultAdd = new AddOperation(gpuConstructor);
const defaultSubtract = new SubtractOperation(gpuConstructor);
const defaultMultiply = new MultiplyOperation(gpuConstructor);
const defaultDivide = new DivideOperation(gpuConstructor);
const defaultPower = new PowerOperation(gpuConstructor);

//Reductions
const defaultMean = new ReduceMeanOperation(gpuConstructor);
const defaultMeanSquare = new ReduceMeanSquareOperation(gpuConstructor);
const defaultSumSquare = new SumSquareOperation(gpuConstructor);
const defaultSum = new SumOperation(gpuConstructor);
const defaultProduct = new ProductOperation(gpuConstructor);
const defaultMax = new MaxOperation(gpuConstructor);
const defaultMin = new MinOperation(gpuConstructor);

//Util
const defaultConcat = new ConcatOperation(gpuConstructor);
const defaultCopy = new CopyOperation(gpuConstructor);
const defaultExpand = new ExpandOperation(gpuConstructor);
const defaultGather = new GatherOperation(gpuConstructor);
const defaultTranspose = new TransposeOperation(gpuConstructor);
const defaultRepeat = new RepeatOperation(gpuConstructor);
const defaultSlice = new SliceOperation(gpuConstructor);
const defaultNormalize = new NormalizeOperation(gpuConstructor);