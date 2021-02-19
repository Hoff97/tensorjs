import Tensor, {Activation, PadMode, Precision} from '../../types';

import {compareShapes, getSize} from '../../util/shape';

import {MatMulOperation} from '../../ops/gpu/matMul/matmul';
import {defaultAllocator, gl} from './gl';
import {MemoryEntry} from './memory';
import {CPUTensor} from '../cpu/tensor';
import REGL from 'regl';
import {ExpOperation} from '../../ops/gpu/unary/exp';
import {GPUTensorConstructor, GPUTensorI} from './interface';
import {ConvBiasOperation, ConvOperation} from '../../ops/gpu/conv/conv';
import {AbsOperation} from '../../ops/gpu/unary/abs';
import {AddOperation} from '../../ops/gpu/binary/add';
import {MultiplyOperation} from '../../ops/gpu/binary/multiply';
import {SubtractOperation} from '../../ops/gpu/binary/subtract';
import {DivideOperation} from '../../ops/gpu/binary/divide';
import {AveragePoolOperation} from '../../ops/gpu/conv/averagePool';
import {ReduceMeanOperation} from '../../ops/gpu/pool/reduceMean';
import {ReduceMeanSquareOperation} from '../../ops/gpu/pool/reduceMeanSquare';
import {SumSquareOperation} from '../../ops/gpu/pool/sumSquare';
import {SumOperation} from '../../ops/gpu/pool/sum';
import {ProductOperation} from '../../ops/gpu/pool/product';
import {MaxOperation} from '../../ops/gpu/pool/max';
import {MinOperation} from '../../ops/gpu/pool/min';
import {CeilOperation} from '../../ops/gpu/unary/ceil';
import {ClipOperation} from '../../ops/gpu/unary/clip';
import {FloorOperation} from '../../ops/gpu/unary/floor';
import {ConcatOperation} from '../../ops/gpu/util/concat';
import {CopyOperation} from '../../ops/gpu/util/copy';
import {ExpandOperation} from '../../ops/gpu/util/expand';
import {GatherOperation} from '../../ops/gpu/util/gather';
import {GemmCOperation, GemmOperation} from '../../ops/gpu/matMul/gemm';
import {PowerOperation} from '../../ops/gpu/binary/power';
import {SqrtOperation} from '../../ops/gpu/unary/sqrt';
import {LogOperation} from '../../ops/gpu/unary/log';
import {TransposeOperation} from '../../ops/gpu/util/transpose';
import {RepeatOperation} from '../../ops/gpu/util/repeat';
import {PadOperation} from '../../ops/gpu/conv/pad';
import {SliceOperation} from '../../ops/gpu/util/slice';
import {UpsampleOperation} from '../../ops/gpu/conv/upsample';
import {NormalizeOperation} from '../../ops/gpu/conv/normalize';
import {Dispatcher} from '../../ops/gpu/dispatcher';
import {SignOperation} from '../../ops/gpu/unary/sign';
import {NegateOperation} from '../../ops/gpu/unary/negate';
import {ClipBackwardOperation} from '../../ops/gpu/util/clipBackward';
import {ConvTransposeOperation} from '../../ops/gpu/conv/convTranspose';
import {SigmoidOperation} from '../../ops/gpu/unary/sigmoid';
import {AddMultiplyScalarOperation} from '../../ops/gpu/unary/addMultiplyScalar';
import {SetValuesOperation} from '../../ops/gpu/util/setValues';
import {
  ASinOperation,
  SinHOperation,
  SinOperation,
} from '../../ops/gpu/unary/sin';
import {
  ACosOperation,
  CosHOperation,
  CosOperation,
} from '../../ops/gpu/unary/cos';
import {
  ATanOperation,
  TanHOperation,
  TanOperation,
} from '../../ops/gpu/unary/tan';
import {ReduceLogSumOperation} from '../../ops/gpu/pool/reduceLogSum';
import {ReduceLogSumExpOperation} from '../../ops/gpu/pool/reduceLogSumExp';
import {HardSigmoidOperation} from '../../ops/gpu/unary/hardSigmoid';
import {PowerScalarOperation} from '../../ops/gpu/unary/powerScalar';
import {RoundOperation} from '../../ops/gpu/unary/round';

export class GPUTensor extends Tensor implements GPUTensorI {
  static range(
    start: number,
    limit: number,
    delta: number,
    precision?: Precision
  ) {
    const size = Math.max(Math.ceil((limit - start) / delta), 0);
    const values = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      values[i] = start + i * delta;
    }
    return new GPUTensor(values, [size], precision || 32);
  }

  public memory: MemoryEntry;

  public size: number;

  public deleted = false;

  constructor(
    values: Float32Array | MemoryEntry,
    public shape: readonly number[],
    public precision: Precision
  ) {
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
      format: 'rgba',
      type: precision === 32 ? 'float' : 'half float',
    });

    const memory = defaultAllocator.allocateFramebuffer(texture, precision);

    const width = texture.width;
    const height = texture.height;

    return new GPUTensor(memory, [height, width, 4], precision);
  }

  getValues(): Promise<Float32Array> {
    return new Promise(resolve => {
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

  constantLike(value: number): Tensor {
    return new GPUTensor(
      new Float32Array(this.size).fill(value),
      this.shape,
      this.precision
    );
  }

  singleConstant(value: number): Tensor {
    return new GPUTensor(new Float32Array([value]), [1], this.precision);
  }

  delete(): void {
    if (!this.deleted) {
      this.deleted = true;
      defaultAllocator.deallocate(this.memory);
      //@ts-ignore
      this.memory = undefined;
    }
  }

  copy(precision?: Precision): Tensor {
    return defaultCopyD.calc(
      {input: this},
      precision ? precision : this.precision
    ) as GPUTensor;
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

  sin(): Tensor {
    return defaultSinD.calc({input: this}, this.precision) as GPUTensor;
  }

  cos(): Tensor {
    return defaultCosD.calc({input: this}, this.precision) as GPUTensor;
  }

  tan(): Tensor {
    return defaultTanD.calc({input: this}, this.precision) as GPUTensor;
  }

  asin(): Tensor {
    return defaultASinD.calc({input: this}, this.precision) as GPUTensor;
  }

  acos(): Tensor {
    return defaultACosD.calc({input: this}, this.precision) as GPUTensor;
  }

  atan(): Tensor {
    return defaultATanD.calc({input: this}, this.precision) as GPUTensor;
  }

  sinh(): Tensor {
    return defaultSinHD.calc({input: this}, this.precision) as GPUTensor;
  }

  cosh(): Tensor {
    return defaultCosHD.calc({input: this}, this.precision) as GPUTensor;
  }

  tanh(): Tensor {
    return defaultTanHD.calc({input: this}, this.precision) as GPUTensor;
  }

  asinh(): Tensor {
    throw new Error('Method not implemented');
  }

  acosh(): Tensor {
    throw new Error('Method not implemented');
  }

  atanh(): Tensor {
    throw new Error('Method not implemented');
  }

  sigmoid(): Tensor {
    return defaultSigmoidD.calc({input: this}, this.precision) as GPUTensor;
  }

  hardSigmoid(alpha: number, beta: number): Tensor {
    return defaultHardSigmoidD.calc(
      {input: this, alpha, beta},
      this.precision
    ) as GPUTensor;
  }

  floor(): Tensor {
    return defaultFloorD.calc({input: this}, this.precision) as GPUTensor;
  }

  ceil(): Tensor {
    return defaultCeilD.calc({input: this}, this.precision) as GPUTensor;
  }

  round(): Tensor {
    return defaultRoundD.calc({input: this}, this.precision) as GPUTensor;
  }

  negate(): Tensor {
    return defaultNegateD.calc({input: this}, this.precision) as GPUTensor;
  }

  addMultiplyScalar(factor: number, add: number): Tensor {
    return defaultAddMultiplyScalarD.calc(
      {input: this, factor, add},
      this.precision
    ) as GPUTensor;
  }

  powerScalar(power: number, factor: number): Tensor {
    return defaultPowerScalarD.calc(
      {input: this, factor, power},
      this.precision
    ) as GPUTensor;
  }

  sign(): Tensor {
    return defaultSignD.calc({input: this}, this.precision) as GPUTensor;
  }

  setValues(values: Tensor, starts: number[]): Tensor {
    if (!(values instanceof GPUTensor)) {
      throw new Error('Can only set GPU values to GPU tensor');
    }
    return defaultSetValuesD.calc(
      {A: this, Values: values, starts},
      this.precision
    ) as GPUTensor;
  }

  add_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only add GPU tensor to GPU tensor');
    }
    return defaultAddD.calc(
      {A: th, B: tensor, outputShape: resultShape, alpha, beta},
      this.precision
    ) as GPUTensor;
  }

  subtract_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only subtract GPU tensor from GPU tensor');
    }
    return defaultSubtractD.calc(
      {A: th, B: tensor, outputShape: resultShape, alpha, beta},
      this.precision
    ) as GPUTensor;
  }

  multiply_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[],
    alpha: number
  ): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only multiply GPU tensor with GPU tensor');
    }
    return defaultMultiplyD.calc(
      {A: th, B: tensor, outputShape: resultShape, alpha},
      this.precision
    ) as GPUTensor;
  }

  divide_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[],
    alpha: number
  ): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only divide GPU tensor by GPU tensor');
    }
    return defaultDivideD.calc(
      {A: th, B: tensor, outputShape: resultShape, alpha},
      this.precision
    ) as GPUTensor;
  }

  power_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[]
  ): Tensor {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only take GPU tensor to power of GPU tensor');
    }
    return defaultPowerD.calc(
      {A: th, B: tensor, outputShape: resultShape},
      this.precision
    ) as GPUTensor;
  }

  matMul(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only matrix multiply GPU tensor to GPU tensor');
    }
    return defaultMatMulD.calc(
      {A: this, B: tensor},
      this.precision
    ) as GPUTensor;
  }

  gemm_impl(
    b: Tensor,
    aTranspose: boolean,
    bTranspose: boolean,
    alpha: number,
    beta: number,
    c?: Tensor
  ): Tensor {
    if (
      !(b instanceof GPUTensor && (c === undefined || c instanceof GPUTensor))
    ) {
      throw new Error('Can only do gemm with CPU tensors');
    }
    if (c === undefined) {
      return defaultGemmD.calc(
        {a: this, b, aTranspose, bTranspose, alpha, beta},
        this.precision
      ) as GPUTensor;
    } else {
      return defaultGemmCD.calc(
        {a: this, b, c: c as GPUTensor, aTranspose, bTranspose, alpha, beta},
        this.precision
      ) as GPUTensor;
    }
  }

  sum_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultSumD.calc(
      {X: this, axes, keepDims},
      this.precision
    ) as GPUTensor;
  }

  sumSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultSumSquareD.calc(
      {X: this, axes, keepDims},
      this.precision
    ) as GPUTensor;
  }

  reduceMean_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultMeanD.calc(
      {X: this, axes, keepDims},
      this.precision
    ) as GPUTensor;
  }

  reduceMeanSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultMeanSquareD.calc(
      {X: this, axes, keepDims},
      this.precision
    ) as GPUTensor;
  }

  protected reduceLogSum_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultLogSumD.calc(
      {X: this, axes, keepDims},
      this.precision
    ) as GPUTensor;
  }

  protected reduceLogSumExp_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultLogSumExpD.calc(
      {X: this, axes, keepDims},
      this.precision
    ) as GPUTensor;
  }

  product_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultProductD.calc(
      {X: this, axes, keepDims},
      this.precision
    ) as GPUTensor;
  }

  max_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultMaxD.calc(
      {X: this, axes, keepDims},
      this.precision
    ) as GPUTensor;
  }

  min_impl(axes: number[], keepDims: boolean): Tensor {
    return defaultMinD.calc(
      {X: this, axes, keepDims},
      this.precision
    ) as GPUTensor;
  }

  conv_impl(
    kernel: Tensor,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[],
    activation?: Activation,
    bias?: Tensor
  ): Tensor {
    if (
      !(kernel instanceof GPUTensor) ||
      (bias !== undefined && !(bias instanceof GPUTensor))
    ) {
      throw new Error('Can only do convolution of GPU tensor with GPU tensor');
    }

    if (bias === undefined) {
      return defaultConvD.calc(
        {
          X: this,
          W: kernel,
          pads,
          dilations,
          strides,
          activation,
        },
        this.precision
      ) as GPUTensor;
    } else {
      return defaultConvBiasD.calc(
        {
          X: this,
          W: kernel,
          B: bias as GPUTensor,
          pads,
          dilations,
          strides,
          activation,
        },
        this.precision
      ) as GPUTensor;
    }
  }

  protected convTranspose_impl(
    kernel: Tensor,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[]
  ): Tensor {
    if (!(kernel instanceof GPUTensor)) {
      throw new Error(
        'Can only do transpose convolution of GPU tensor with GPU tensor'
      );
    }

    return defaultConvTransposeD.calc(
      {
        X: this,
        W: kernel,
        pads,
        dilations,
        strides,
      },
      this.precision
    ) as GPUTensor;
  }

  averagePool_impl(
    kernelShape: number[],
    pads: number[],
    strides: number[],
    includePad: boolean
  ): Tensor {
    return defaultAveragePoolD.calc(
      {
        X: this,
        includePad,
        kernelShape,
        pads,
        strides,
      },
      this.precision
    ) as GPUTensor;
  }

  reshape_impl(shape: number[], _copy: boolean): Tensor {
    if (_copy) {
      return defaultCopyD.calc(
        {input: this, outputShape: shape},
        this.precision
      ) as GPUTensor;
    } else {
      return new GPUTensor(this.memory, shape, this.precision);
    }
  }

  concat(tensor: Tensor, axis: number): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only concat GPU tensor to GPU tensor');
    }
    if (axis < 0) {
      axis += this.shape.length;
    }
    return defaultConcatD.calc(
      {A: this, B: tensor, axis},
      this.precision
    ) as GPUTensor;
  }

  transpose_impl(permutation: number[]): Tensor {
    return defaultTransposeD.calc(
      {A: this, permutation},
      this.precision
    ) as GPUTensor;
  }

  clip(min?: number, max?: number): Tensor {
    return defaultClipD.calc(
      {input: this, minVal: min, maxVal: max},
      this.precision
    ) as GPUTensor;
  }

  clipBackward(grad: Tensor, min?: number, max?: number): Tensor {
    return defaultClipBackwardD.calc(
      {input: this, minVal: min, maxVal: max, grad},
      this.precision
    ) as GPUTensor;
  }

  repeat(repeats: number[]): Tensor {
    return defaultRepeatD.calc({A: this, repeats}, this.precision) as GPUTensor;
  }

  expand(shape: readonly number[]): Tensor {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [_shape, _o, resultShape] = this.alignShapes(this.shape, shape);
    if (compareShapes(this.shape, resultShape)) {
      return this.copy();
    }
    return defaultExpandD.calc(
      {
        input: this.reshape(_shape, false) as GPUTensor,
        outputShape: resultShape,
      },
      this.precision
    ) as GPUTensor;
  }

  pad_impl(pads: number[], mode: PadMode, value: number): Tensor {
    return defaultPadD.calc(
      {input: this, pads, mode, value},
      this.precision
    ) as GPUTensor;
  }

  gather(axis: number, indices: CPUTensor): Tensor {
    return defaultGatherD.calc(
      {X: this, axis, indices},
      this.precision
    ) as GPUTensor;
  }

  slice_impl(
    starts: number[],
    ends: number[],
    axes: number[],
    steps: number[]
  ): Tensor {
    return defaultSliceD.calc(
      {X: this, starts, ends, axes, steps},
      this.precision
    ) as GPUTensor;
  }

  upsample(scales: number[]): Tensor {
    return defaultUpsampleD.calc(
      {X: this, scales},
      this.precision
    ) as GPUTensor;
  }

  normalize(
    mean: Tensor,
    variance: Tensor,
    epsilon: number,
    scale: Tensor,
    bias: Tensor
  ): Tensor {
    if (
      !(mean instanceof GPUTensor) ||
      !(variance instanceof GPUTensor) ||
      !(scale instanceof GPUTensor) ||
      !(bias instanceof GPUTensor)
    ) {
      throw new Error('Can only normalize with CPU tensors');
    }

    return defaultNormalizeD.calc(
      {
        X: this,
        Mean: mean,
        Variance: variance,
        Scale: scale,
        Bias: bias,
        epsilon,
      },
      this.precision
    ) as GPUTensor;
  }
}

export const gpuConstructor: GPUTensorConstructor<GPUTensor> = (
  a: MemoryEntry,
  b: readonly number[],
  precision: Precision
) => new GPUTensor(a, b, precision);

const defaultMatMulD = new Dispatcher(
  () => new MatMulOperation(gpuConstructor)
);

const defaultGemmD = new Dispatcher(() => new GemmOperation(gpuConstructor));
const defaultGemmCD = new Dispatcher(() => new GemmCOperation(gpuConstructor));

//Unary operations
const defaultExpD = new Dispatcher(() => new ExpOperation(gpuConstructor));
const defaultAbsD = new Dispatcher(() => new AbsOperation(gpuConstructor));
const defaultSinD = new Dispatcher(() => new SinOperation(gpuConstructor));
const defaultCosD = new Dispatcher(() => new CosOperation(gpuConstructor));
const defaultTanD = new Dispatcher(() => new TanOperation(gpuConstructor));
const defaultASinD = new Dispatcher(() => new ASinOperation(gpuConstructor));
const defaultACosD = new Dispatcher(() => new ACosOperation(gpuConstructor));
const defaultATanD = new Dispatcher(() => new ATanOperation(gpuConstructor));
const defaultSinHD = new Dispatcher(() => new SinHOperation(gpuConstructor));
const defaultCosHD = new Dispatcher(() => new CosHOperation(gpuConstructor));
const defaultTanHD = new Dispatcher(() => new TanHOperation(gpuConstructor));
const defaultSigmoidD = new Dispatcher(
  () => new SigmoidOperation(gpuConstructor)
);
const defaultHardSigmoidD = new Dispatcher(
  () => new HardSigmoidOperation(gpuConstructor)
);
const defaultCeilD = new Dispatcher(() => new CeilOperation(gpuConstructor));
const defaultFloorD = new Dispatcher(() => new FloorOperation(gpuConstructor));
const defaultRoundD = new Dispatcher(() => new RoundOperation(gpuConstructor));
const defaultClipD = new Dispatcher(() => new ClipOperation(gpuConstructor));
const defaultClipBackwardD = new Dispatcher(
  () => new ClipBackwardOperation(gpuConstructor)
);
const defaultSqrtD = new Dispatcher(() => new SqrtOperation(gpuConstructor));
const defaultLogD = new Dispatcher(() => new LogOperation(gpuConstructor));
const defaultNegateD = new Dispatcher(
  () => new NegateOperation(gpuConstructor)
);
const defaultAddMultiplyScalarD = new Dispatcher(
  () => new AddMultiplyScalarOperation(gpuConstructor)
);
const defaultPowerScalarD = new Dispatcher(
  () => new PowerScalarOperation(gpuConstructor)
);
const defaultSignD = new Dispatcher(() => new SignOperation(gpuConstructor));

//Convolutions
const defaultConvD = new Dispatcher(() => new ConvOperation(gpuConstructor));
const defaultAveragePoolD = new Dispatcher(
  () => new AveragePoolOperation(gpuConstructor)
);
const defaultConvBiasD = new Dispatcher(
  () => new ConvBiasOperation(gpuConstructor)
);
const defaultConvTransposeD = new Dispatcher(
  () => new ConvTransposeOperation(gpuConstructor)
);
const defaultPadD = new Dispatcher(() => new PadOperation(gpuConstructor));
const defaultUpsampleD = new Dispatcher(
  () => new UpsampleOperation(gpuConstructor)
);

//Binary operations
const defaultAddD = new Dispatcher(() => new AddOperation(gpuConstructor));
const defaultSubtractD = new Dispatcher(
  () => new SubtractOperation(gpuConstructor)
);
const defaultMultiplyD = new Dispatcher(
  () => new MultiplyOperation(gpuConstructor)
);
const defaultDivideD = new Dispatcher(
  () => new DivideOperation(gpuConstructor)
);
const defaultPowerD = new Dispatcher(() => new PowerOperation(gpuConstructor));

//Reductions
const defaultMeanD = new Dispatcher(
  () => new ReduceMeanOperation(gpuConstructor)
);
const defaultMeanSquareD = new Dispatcher(
  () => new ReduceMeanSquareOperation(gpuConstructor)
);
const defaultSumSquareD = new Dispatcher(
  () => new SumSquareOperation(gpuConstructor)
);
const defaultSumD = new Dispatcher(() => new SumOperation(gpuConstructor));
const defaultProductD = new Dispatcher(
  () => new ProductOperation(gpuConstructor)
);
const defaultMaxD = new Dispatcher(() => new MaxOperation(gpuConstructor));
const defaultMinD = new Dispatcher(() => new MinOperation(gpuConstructor));
const defaultLogSumD = new Dispatcher(
  () => new ReduceLogSumOperation(gpuConstructor)
);
const defaultLogSumExpD = new Dispatcher(
  () => new ReduceLogSumExpOperation(gpuConstructor)
);

//Util
const defaultConcatD = new Dispatcher(
  () => new ConcatOperation(gpuConstructor)
);
const defaultSetValuesD = new Dispatcher(
  () => new SetValuesOperation(gpuConstructor)
);
const defaultCopyD = new Dispatcher(() => new CopyOperation(gpuConstructor));
const defaultExpandD = new Dispatcher(
  () => new ExpandOperation(gpuConstructor)
);
const defaultGatherD = new Dispatcher(
  () => new GatherOperation(gpuConstructor)
);
const defaultTransposeD = new Dispatcher(
  () => new TransposeOperation(gpuConstructor)
);
const defaultRepeatD = new Dispatcher(
  () => new RepeatOperation(gpuConstructor)
);
const defaultSliceD = new Dispatcher(() => new SliceOperation(gpuConstructor));
const defaultNormalizeD = new Dispatcher(
  () => new NormalizeOperation(gpuConstructor)
);
