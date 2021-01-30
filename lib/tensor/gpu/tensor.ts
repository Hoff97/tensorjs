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
import { CopyInfo, CopyInput, CopyOperation } from '../../ops/gpu/util/copy';
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
import { Dispatcher } from '../../ops/gpu/dispatcher';


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

  delete(): void {
    this.deleted = true;
    defaultAllocator.deallocate(this.memory);
    this.memory = undefined;
  }

  copy(precision?: Precision): Tensor {
    return defaultCopyD.calc({input: this}, precision ? precision : this.precision) as GPUTensor;
  }

  exp(): Tensor {
    return defaultExpD.calc({input: this}, this.precision) as GPUTensor;
  }

  log(): Tensor {
    return defaultLogD.calc({input: this}, this.precision) as GPUTensor;
  }

  sqrt(): Tensor {
    return defaultSqrtD.calc({input: this}, this.precision) as GPUTensor;
  }

  abs(): Tensor {
    return defaultAbsD.calc({input: this}, this.precision) as GPUTensor;
  }

  floor(): Tensor {
    return defaultFloorD.calc({input: this}, this.precision) as GPUTensor;
  }

  ceil(): Tensor {
    return defaultCeilD.calc({input: this}, this.precision) as GPUTensor
  }

  add_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only add GPU tensor to GPU tensor');
    }
    return defaultAddD.calc({A: th, B: tensor, outputShape: resultShape}, this.precision) as GPUTensor;
  }

  subtract_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only subtract GPU tensor from GPU tensor');
    }
    return defaultSubtractD.calc({A: th, B: tensor, outputShape: resultShape}, this.precision) as GPUTensor;
  }

  multiply_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only multiply GPU tensor with GPU tensor');
    }
    return defaultMultiplyD.calc({A: th, B: tensor, outputShape: resultShape}, this.precision) as GPUTensor;
  }

  divide_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only divide GPU tensor by GPU tensor');
    }
    return defaultDivideD.calc({A: th, B: tensor, outputShape: resultShape}, this.precision) as GPUTensor;
  }

  power_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only take GPU tensor to power of GPU tensor');
    }
    return defaultPowerD.calc({A: th, B: tensor, outputShape: resultShape}, this.precision) as GPUTensor;
  }

  matMul(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only matrix multiply GPU tensor to GPU tensor');
    }
    return defaultMatMulD.calc({A: this, B: tensor}, this.precision) as GPUTensor;
  }

  gemm_impl(b: Tensor, aTranspose: boolean, bTranspose: boolean, alpha: number, beta: number, c?: Tensor): Tensor {
    if (!(b instanceof GPUTensor && (c === undefined || c instanceof GPUTensor))) {
      throw new Error('Can only do gemm with CPU tensors');
    }
    if (c === undefined) {
      return defaultGemmD.calc({a: this, b, aTranspose, bTranspose, alpha, beta}, this.precision) as GPUTensor;
    } else {
      return defaultGemmCD.calc({a: this, b, c: c as GPUTensor, aTranspose, bTranspose, alpha, beta}, this.precision) as GPUTensor;
    }
  }

  sum_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultSumD.calc({X: this, axes, keepDims}, this.precision) as GPUTensor;
  }

  sumSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultSumSquareD.calc({X: this, axes, keepDims}, this.precision) as GPUTensor;
  }

  reduceMean_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultMeanD.calc({X: this, axes, keepDims}, this.precision) as GPUTensor;
  }

  reduceMeanSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultMeanSquareD.calc({X: this, axes, keepDims}, this.precision) as GPUTensor;
  }

  product_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultProductD.calc({X: this, axes, keepDims}, this.precision) as GPUTensor;
  }

  max_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultMaxD.calc({X: this, axes, keepDims}, this.precision) as GPUTensor;
  }

  min_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultMinD.calc({X: this, axes, keepDims}, this.precision) as GPUTensor;
  }

  conv_impl(kernel: Tensor, dilations: number[], group: number, pads: number[], strides: number[], bias?: Tensor, activation?: Activation): Tensor {
    if (!(kernel instanceof GPUTensor) || (bias !== undefined && !(bias instanceof GPUTensor))) {
      throw new Error('Can only do convolution of GPU tensor with GPU tensor');
    }

    if (bias === undefined) {
      return defaultConvD.calc({
        X: this,
        W: kernel,
        pads, dilations, strides,
        activation
      }, this.precision) as GPUTensor;
    } else {
      return defaultConvBiasD.calc({
        X: this,
        W: kernel,
        B: bias as GPUTensor,
        pads, dilations, strides,
        activation
      }, this.precision) as GPUTensor;
    }
  }

  averagePool_impl(kernelShape: number[], pads: number[], strides: number[], includePad: boolean): Tensor {
    return defaultAveragePoolD.calc({
      X: this,
      includePad,
      kernelShape,
      pads,
      strides
    }, this.precision) as GPUTensor;
  }

  reshape_impl(shape: number[], _copy: boolean): Tensor {
    if (_copy) {
      return defaultCopyD.calc({input: this, outputShape: shape}, this.precision) as GPUTensor;
    } else {
      return new GPUTensor(this.memory, shape, this.precision);
    }
  }

  concat(tensor: Tensor, axis: number): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only concat GPU tensor to GPU tensor');
    }
    return defaultConcatD.calc({A: this, B: tensor, axis}, this.precision) as GPUTensor;
  }

  transpose_impl(permutation: number[]): Tensor {
    return defaultTransposeD.calc({A: this, permutation}, this.precision) as GPUTensor;
  }

  clip(min?: number, max?: number): Tensor {
    return defaultClipD.calc({input: this, minVal: min, maxVal: max}, this.precision) as GPUTensor;
  }

  repeat(repeats: number[]): Tensor {
    return defaultRepeatD.calc({A: this, repeats}, this.precision) as GPUTensor;
  }

  expand(shape: number[]): Tensor {
    const [_shape, goal, resultShape] = this.alignShapes(this.shape, shape);
    if (compareShapes(this.shape, resultShape)) {
      return this.copy();
    }
    return defaultExpandD.calc({input: this.reshape(_shape, false) as GPUTensor, outputShape: resultShape}, this.precision) as GPUTensor;
  }

  pad_impl(pads: number[], mode: PadMode, value: number): Tensor {
    return defaultPadD.calc({input: this, pads, mode, value}, this.precision) as GPUTensor;
  }

  gather(axis: number, indices: CPUTensor): Tensor {
    return defaultGatherD.calc({X: this, axis, indices}, this.precision) as GPUTensor;
  }

  slice_impl(starts: number[], ends: number[], axes: number[]): Tensor {
    return defaultSliceD.calc({X: this, starts, ends, axes}, this.precision) as GPUTensor;
  }

  upsample(scales: number[]): Tensor {
    return defaultUpsampleD.calc({X: this, scales}, this.precision) as GPUTensor;
  }

  normalize(mean: Tensor, variance: Tensor, epsilon: number, scale: Tensor, bias: Tensor): Tensor {
    if (!(mean instanceof GPUTensor) || !(variance instanceof GPUTensor) || !(scale instanceof GPUTensor) || !(bias instanceof GPUTensor)) {
      throw new Error('Can only normalize with CPU tensors');
    }

    return defaultNormalizeD.calc({
      X: this,
      Mean: mean,
      Variance: variance,
      Scale: scale,
      Bias: bias,
      epsilon
    }, this.precision) as GPUTensor;
  }
}

export const gpuConstructor: GPUTensorConstructor<GPUTensor> = (a: MemoryEntry,b: readonly number[], precision: Precision) => new GPUTensor(a,b,precision);

const defaultMatMulD = new Dispatcher(() => new MatMulOperation(gpuConstructor));
//const defaultMatMul = new MatMulOperation(gpuConstructor);
const defaultGemmD = new Dispatcher(() => new GemmOperation(gpuConstructor));
//const defaultGemm = new GemmOperation(gpuConstructor);
const defaultGemmCD = new Dispatcher(() => new GemmCOperation(gpuConstructor));
//const defaultGemmC = new GemmCOperation(gpuConstructor);

//Unary operations
const defaultExpD = new Dispatcher(() => new ExpOperation(gpuConstructor));
//const defaultExp = new ExpOperation(gpuConstructor);
const defaultAbsD = new Dispatcher(() => new AbsOperation(gpuConstructor));
//const defaultAbs = new AbsOperation(gpuConstructor);
const defaultCeilD = new Dispatcher(() => new CeilOperation(gpuConstructor));
//const defaultCeil = new CeilOperation(gpuConstructor);
const defaultFloorD = new Dispatcher(() => new FloorOperation(gpuConstructor));
//const defaultFloor = new FloorOperation(gpuConstructor);
const defaultClipD = new Dispatcher(() => new ClipOperation(gpuConstructor));
//const defaultClip = new ClipOperation(gpuConstructor);
const defaultSqrtD = new Dispatcher(() => new SqrtOperation(gpuConstructor));
//const defaultSqrt = new SqrtOperation(gpuConstructor);
const defaultLogD = new Dispatcher(() => new LogOperation(gpuConstructor));
//const defaultLog = new LogOperation(gpuConstructor);

//Convolutions
const defaultConvD = new Dispatcher(() => new ConvOperation(gpuConstructor));
//const defaultConv = new ConvOperation(gpuConstructor);
const defaultAveragePoolD = new Dispatcher(() => new AveragePoolOperation(gpuConstructor));
//const defaultAveragePool = new AveragePoolOperation(gpuConstructor);
const defaultConvBiasD = new Dispatcher(() => new ConvBiasOperation(gpuConstructor));
//const defaultConvBias = new ConvBiasOperation(gpuConstructor);
const defaultPadD = new Dispatcher(() => new PadOperation(gpuConstructor));
//const defaultPad = new PadOperation(gpuConstructor);
const defaultUpsampleD = new Dispatcher(() => new UpsampleOperation(gpuConstructor));
//const defaultUpsample = new UpsampleOperation(gpuConstructor);

//Binary operations
const defaultAddD = new Dispatcher(() => new AddOperation(gpuConstructor));
//const defaultAdd = new AddOperation(gpuConstructor);
const defaultSubtractD = new Dispatcher(() => new SubtractOperation(gpuConstructor));
//const defaultSubtract = new SubtractOperation(gpuConstructor);
const defaultMultiplyD = new Dispatcher(() => new MultiplyOperation(gpuConstructor));
//const defaultMultiply = new MultiplyOperation(gpuConstructor);
const defaultDivideD = new Dispatcher(() => new DivideOperation(gpuConstructor));
//const defaultDivide = new DivideOperation(gpuConstructor);
const defaultPowerD = new Dispatcher(() => new PowerOperation(gpuConstructor));
//const defaultPower = new PowerOperation(gpuConstructor);

//Reductions
const defaultMeanD = new Dispatcher(() => new ReduceMeanOperation(gpuConstructor));
//const defaultMean = new ReduceMeanOperation(gpuConstructor);
const defaultMeanSquareD = new Dispatcher(() => new ReduceMeanSquareOperation(gpuConstructor));
//const defaultMeanSquare = new ReduceMeanSquareOperation(gpuConstructor);
const defaultSumSquareD = new Dispatcher(() => new SumSquareOperation(gpuConstructor));
//const defaultSumSquare = new SumSquareOperation(gpuConstructor);
const defaultSumD = new Dispatcher(() => new SumOperation(gpuConstructor));
//const defaultSum = new SumOperation(gpuConstructor);
const defaultProductD = new Dispatcher(() => new ProductOperation(gpuConstructor));
//const defaultProduct = new ProductOperation(gpuConstructor);
const defaultMaxD = new Dispatcher(() => new MaxOperation(gpuConstructor));
//const defaultMax = new MaxOperation(gpuConstructor);
const defaultMinD = new Dispatcher(() => new MinOperation(gpuConstructor));
//const defaultMin = new MinOperation(gpuConstructor);

//Util
const defaultConcatD = new Dispatcher(() => new ConcatOperation(gpuConstructor));
//const defaultConcat = new ConcatOperation(gpuConstructor);
const defaultCopyD = new Dispatcher(() => new CopyOperation(gpuConstructor));
//const defaultCopy = new CopyOperation(gpuConstructor);
const defaultExpandD = new Dispatcher(() => new ExpandOperation(gpuConstructor));
//const defaultExpand = new ExpandOperation(gpuConstructor);
const defaultGatherD = new Dispatcher(() => new GatherOperation(gpuConstructor));
//const defaultGather = new GatherOperation(gpuConstructor);
const defaultTransposeD = new Dispatcher(() => new TransposeOperation(gpuConstructor));
//const defaultTranspose = new TransposeOperation(gpuConstructor);
const defaultRepeatD = new Dispatcher(() => new RepeatOperation(gpuConstructor));
//const defaultRepeat = new RepeatOperation(gpuConstructor);
const defaultSliceD = new Dispatcher(() => new SliceOperation(gpuConstructor));
//const defaultSlice = new SliceOperation(gpuConstructor);
const defaultNormalizeD = new Dispatcher(() => new NormalizeOperation(gpuConstructor));
//const defaultNormalize = new NormalizeOperation(gpuConstructor);