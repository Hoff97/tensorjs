import Tensor, {
  Activation,
  DType,
  PadMode,
  TensorValues,
  tensorValuesConstructor,
} from '../../types';

import {compareShapes, getSize} from '../../util/shape';

import {MatMulOperation} from '../../ops/gpu/matMul/matmul';
import {defaultAllocator, gl} from './gl';
import {MemoryEntry} from './memory';
import {CPUTensor} from '../cpu/tensor';
import REGL from 'regl';
import {ExpOperation} from '../../ops/gpu/unary/exp';
import {DTypeGpu, GPUTensorConstructor, GPUTensorI} from './interface';
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
import {ArgMaxOperation} from '../../ops/gpu/pool/argMax';
import {ArgMinOperation} from '../../ops/gpu/pool/argMin';

export class GPUTensor<DTpe extends DTypeGpu = 'float32'>
  extends Tensor<DTpe>
  implements GPUTensorI {
  static range(start: number, limit: number, delta: number, dtype?: DTypeGpu) {
    const size = Math.max(Math.ceil((limit - start) / delta), 0);
    const values = new Array(size);
    for (let i = 0; i < size; i++) {
      values[i] = start + i * delta;
    }
    return new GPUTensor(values, [size], dtype);
  }

  public memory: MemoryEntry;

  public size: number;

  public deleted = false;

  constructor(
    values: number[] | MemoryEntry,
    public shape: readonly number[],
    dtype?: DTpe
  ) {
    super(dtype || ('float32' as any));

    this.size = getSize(shape);

    if (values instanceof Array) {
      this.memory = defaultAllocator.allocateTexture(values, this.dtype);
    } else {
      this.memory = values;
    }
  }

  static fromData(data: REGL.TextureImageData) {
    const texture = gl.texture({
      data: data,
      format: 'rgba',
      type: defaultAllocator.getColorType('float32') as any,
    });

    const memory = defaultAllocator.allocateFramebuffer(texture, 'float32');

    const width = texture.width;
    const height = texture.height;

    return new GPUTensor(memory, [height, width, 4], 'float32');
  }

  cast<DTpe2 extends DType>(dtype: DTpe2): Tensor<DTpe2> {
    if (dtype === 'float64') {
      throw new Error('The WebGL backend does not support float64 tensors');
    }
    return defaultCopyD.calc({input: this}, dtype as any) as any;
  }

  getValues(): Promise<TensorValues[DTpe]> {
    if (this.dtype !== 'float16') {
      return new Promise(resolve => {
        gl({framebuffer: this.memory.frameBuffer})(() => {
          let result = new Float32Array(this.memory.size);
          result = gl.read(result);

          if (this.dtype === 'float32') {
            resolve(result.subarray(0, this.size) as TensorValues[DTpe]);
          } else {
            const arr = new tensorValuesConstructor[this.dtype](
              this.size
            ) as TensorValues[DTpe];
            for (let i = 0; i < this.size; i++) {
              arr[i] = result[i];
            }
            resolve(arr);
          }
        });
      });
    }
    throw new Error('Reading values not supported for data type float16');
  }

  getShape(): readonly number[] {
    return this.shape;
  }

  constantLike(value: number): Tensor<DTpe> {
    return new GPUTensor(
      new Array(this.size).fill(value),
      this.shape,
      this.dtype
    );
  }

  singleConstant(value: number): Tensor<DTpe> {
    return new GPUTensor([value], [1], this.dtype);
  }

  delete(): void {
    if (!this.deleted) {
      this.deleted = true;
      defaultAllocator.deallocate(this.memory);
      //@ts-ignore
      this.memory = undefined;
    }
  }

  copy(): Tensor<DTpe> {
    return defaultCopyD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  exp(): Tensor<DTpe> {
    return defaultExpD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  log(): Tensor<DTpe> {
    return defaultLogD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  sqrt(): Tensor<DTpe> {
    return defaultSqrtD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  abs(): Tensor<DTpe> {
    return defaultAbsD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  sin(): Tensor<DTpe> {
    return defaultSinD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  cos(): Tensor<DTpe> {
    return defaultCosD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  tan(): Tensor<DTpe> {
    return defaultTanD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  asin(): Tensor<DTpe> {
    return defaultASinD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  acos(): Tensor<DTpe> {
    return defaultACosD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  atan(): Tensor<DTpe> {
    return defaultATanD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  sinh(): Tensor<DTpe> {
    return defaultSinHD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  cosh(): Tensor<DTpe> {
    return defaultCosHD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  tanh(): Tensor<DTpe> {
    return defaultTanHD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  asinh(): Tensor<DTpe> {
    throw new Error('Method not implemented');
  }

  acosh(): Tensor<DTpe> {
    throw new Error('Method not implemented');
  }

  atanh(): Tensor<DTpe> {
    throw new Error('Method not implemented');
  }

  sigmoid(): Tensor<DTpe> {
    return defaultSigmoidD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  hardSigmoid(alpha: number, beta: number): Tensor<DTpe> {
    return defaultHardSigmoidD.calc(
      {input: this, alpha, beta},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  floor(): Tensor<DTpe> {
    return defaultFloorD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  ceil(): Tensor<DTpe> {
    return defaultCeilD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  round(): Tensor<DTpe> {
    return defaultRoundD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  negate(): Tensor<DTpe> {
    return defaultNegateD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  addMultiplyScalar(factor: number, add: number): Tensor<DTpe> {
    return defaultAddMultiplyScalarD.calc(
      {input: this, factor, add},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  powerScalar(power: number, factor: number): Tensor<DTpe> {
    return defaultPowerScalarD.calc(
      {input: this, factor, power},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  sign(): Tensor<DTpe> {
    return defaultSignD.calc({input: this}, this.dtype) as GPUTensor<DTpe>;
  }

  setValues(values: Tensor<DTpe>, starts: number[]): Tensor<DTpe> {
    if (!(values instanceof GPUTensor)) {
      throw new Error('Can only set GPU values to GPU tensor');
    }
    return defaultSetValuesD.calc(
      {A: this, Values: values, starts},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  add_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only add GPU tensor to GPU tensor');
    }
    return defaultAddD.calc(
      {A: th, B: tensor, outputShape: resultShape, alpha, beta},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  subtract_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only subtract GPU tensor from GPU tensor');
    }
    return defaultSubtractD.calc(
      {A: th, B: tensor, outputShape: resultShape, alpha, beta},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  multiply_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only multiply GPU tensor with GPU tensor');
    }
    return defaultMultiplyD.calc(
      {A: th, B: tensor, outputShape: resultShape, alpha},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  divide_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only divide GPU tensor by GPU tensor');
    }
    return defaultDivideD.calc(
      {A: th, B: tensor, outputShape: resultShape, alpha},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  power_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[]
  ): Tensor<DTpe> {
    if (!(tensor instanceof GPUTensor) || !(th instanceof GPUTensor)) {
      throw new Error('Can only take GPU tensor to power of GPU tensor');
    }
    return defaultPowerD.calc(
      {A: th, B: tensor, outputShape: resultShape},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  matMul(tensor: Tensor<DTpe>): Tensor<DTpe> {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only matrix multiply GPU tensor to GPU tensor');
    }
    return defaultMatMulD.calc(
      {A: this, B: tensor},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  gemm_impl(
    b: Tensor<DTpe>,
    aTranspose: boolean,
    bTranspose: boolean,
    alpha: number,
    beta: number,
    c?: Tensor<DTpe>
  ): Tensor<DTpe> {
    if (
      !(b instanceof GPUTensor && (c === undefined || c instanceof GPUTensor))
    ) {
      throw new Error('Can only do gemm with CPU tensors');
    }
    if (c === undefined) {
      return defaultGemmD.calc(
        {a: this, b, aTranspose, bTranspose, alpha, beta},
        this.dtype
      ) as GPUTensor<DTpe>;
    } else {
      return defaultGemmCD.calc(
        {
          a: this,
          b,
          c: c as GPUTensor<DTpe>,
          aTranspose,
          bTranspose,
          alpha,
          beta,
        },
        this.dtype
      ) as GPUTensor<DTpe>;
    }
  }

  sum_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return defaultSumD.calc(
      {X: this, axes, keepDims},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  sumSquare_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return defaultSumSquareD.calc(
      {X: this, axes, keepDims},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  reduceMean_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return defaultMeanD.calc(
      {X: this, axes, keepDims},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  reduceMeanSquare_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return defaultMeanSquareD.calc(
      {X: this, axes, keepDims},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  protected reduceLogSum_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return defaultLogSumD.calc(
      {X: this, axes, keepDims},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  protected reduceLogSumExp_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor<DTpe> {
    return defaultLogSumExpD.calc(
      {X: this, axes, keepDims},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  product_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return defaultProductD.calc(
      {X: this, axes, keepDims},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  max_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return defaultMaxD.calc(
      {X: this, axes, keepDims},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  protected argMax_impl(axes: number[], selectLast: boolean): Tensor<'uint32'> {
    return defaultArgMaxD.calc(
      {X: this, axes, selectLast},
      this.dtype
    ) as GPUTensor<'uint32'>;
  }

  protected argMin_impl(axes: number[], selectLast: boolean): Tensor<'uint32'> {
    return defaultArgMinD.calc(
      {X: this, axes, selectLast},
      this.dtype
    ) as GPUTensor<'uint32'>;
  }

  min_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return defaultMinD.calc(
      {X: this, axes, keepDims},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  conv_impl(
    kernel: Tensor<DTpe>,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[],
    activation?: Activation,
    bias?: Tensor<DTpe>
  ): Tensor<DTpe> {
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
        this.dtype
      ) as GPUTensor<DTpe>;
    } else {
      return defaultConvBiasD.calc(
        {
          X: this,
          W: kernel,
          B: bias as GPUTensor<DTpe>,
          pads,
          dilations,
          strides,
          activation,
        },
        this.dtype
      ) as GPUTensor<DTpe>;
    }
  }

  protected convTranspose_impl(
    kernel: Tensor<DTpe>,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[]
  ): Tensor<DTpe> {
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
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  averagePool_impl(
    kernelShape: number[],
    pads: number[],
    strides: number[],
    includePad: boolean
  ): Tensor<DTpe> {
    return defaultAveragePoolD.calc(
      {
        X: this,
        includePad,
        kernelShape,
        pads,
        strides,
      },
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  reshape_impl(shape: number[], _copy: boolean): Tensor<DTpe> {
    if (_copy) {
      return defaultCopyD.calc(
        {input: this, outputShape: shape},
        this.dtype
      ) as GPUTensor<DTpe>;
    } else {
      return new GPUTensor(this.memory, shape, this.dtype);
    }
  }

  concat(tensor: Tensor<DTpe>, axis: number): Tensor<DTpe> {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only concat GPU tensor to GPU tensor');
    }
    if (axis < 0) {
      axis += this.shape.length;
    }
    return defaultConcatD.calc(
      {A: this, B: tensor, axis},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  transpose_impl(permutation: number[]): Tensor<DTpe> {
    return defaultTransposeD.calc(
      {A: this, permutation},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  clip(min?: number, max?: number): Tensor<DTpe> {
    return defaultClipD.calc(
      {input: this, minVal: min, maxVal: max},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  clipBackward(grad: Tensor<DTpe>, min?: number, max?: number): Tensor<DTpe> {
    return defaultClipBackwardD.calc(
      {input: this, minVal: min, maxVal: max, grad},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  repeat(repeats: number[]): Tensor<DTpe> {
    return defaultRepeatD.calc(
      {A: this, repeats},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  expand(shape: readonly number[]): Tensor<DTpe> {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [_shape, _o, resultShape] = this.alignShapes(this.shape, shape);
    if (compareShapes(this.shape, resultShape)) {
      return this.copy();
    }
    return defaultExpandD.calc(
      {
        input: this.reshape(_shape, false) as GPUTensor<DTpe>,
        outputShape: resultShape,
      },
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  pad_impl(pads: number[], mode: PadMode, value: number): Tensor<DTpe> {
    return defaultPadD.calc(
      {input: this, pads, mode, value},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  gather(axis: number, indices: CPUTensor<'uint32'>): Tensor<DTpe> {
    return defaultGatherD.calc(
      {X: this, axis, indices},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  slice_impl(
    starts: number[],
    ends: number[],
    axes: number[],
    steps: number[]
  ): Tensor<DTpe> {
    return defaultSliceD.calc(
      {X: this, starts, ends, axes, steps},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  upsample(scales: number[]): Tensor<DTpe> {
    return defaultUpsampleD.calc(
      {X: this, scales},
      this.dtype
    ) as GPUTensor<DTpe>;
  }

  normalize(
    mean: Tensor<DTpe>,
    variance: Tensor<DTpe>,
    epsilon: number,
    scale: Tensor<DTpe>,
    bias: Tensor<DTpe>
  ): Tensor<DTpe> {
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
      this.dtype
    ) as GPUTensor<DTpe>;
  }
}

export function gpuConstructor<DTpe extends DTypeGpu>(
  a: MemoryEntry,
  b: readonly number[],
  dtype: DTpe
) {
  return new GPUTensor(a, b, dtype);
}

const defaultMatMulD = new Dispatcher(
  (dtype: DTypeGpu) => new MatMulOperation(gpuConstructor, dtype)
);

const defaultGemmD = new Dispatcher(
  (dtype: DTypeGpu) => new GemmOperation(gpuConstructor, dtype)
);
const defaultGemmCD = new Dispatcher(
  (dtype: DTypeGpu) => new GemmCOperation(gpuConstructor, dtype)
);

//Unary operations
const defaultExpD = new Dispatcher(
  (dtype: DTypeGpu) => new ExpOperation(gpuConstructor, dtype)
);
const defaultAbsD = new Dispatcher(
  (dtype: DTypeGpu) => new AbsOperation(gpuConstructor, dtype)
);
const defaultSinD = new Dispatcher(
  (dtype: DTypeGpu) => new SinOperation(gpuConstructor, dtype)
);
const defaultCosD = new Dispatcher(
  (dtype: DTypeGpu) => new CosOperation(gpuConstructor, dtype)
);
const defaultTanD = new Dispatcher(
  (dtype: DTypeGpu) => new TanOperation(gpuConstructor, dtype)
);
const defaultASinD = new Dispatcher(
  (dtype: DTypeGpu) => new ASinOperation(gpuConstructor, dtype)
);
const defaultACosD = new Dispatcher(
  (dtype: DTypeGpu) => new ACosOperation(gpuConstructor, dtype)
);
const defaultATanD = new Dispatcher(
  (dtype: DTypeGpu) => new ATanOperation(gpuConstructor, dtype)
);
const defaultSinHD = new Dispatcher(
  (dtype: DTypeGpu) => new SinHOperation(gpuConstructor, dtype)
);
const defaultCosHD = new Dispatcher(
  (dtype: DTypeGpu) => new CosHOperation(gpuConstructor, dtype)
);
const defaultTanHD = new Dispatcher(
  (dtype: DTypeGpu) => new TanHOperation(gpuConstructor, dtype)
);
const defaultSigmoidD = new Dispatcher(
  (dtype: DTypeGpu) => new SigmoidOperation(gpuConstructor, dtype)
);
const defaultHardSigmoidD = new Dispatcher(
  (dtype: DTypeGpu) => new HardSigmoidOperation(gpuConstructor, dtype)
);
const defaultCeilD = new Dispatcher(
  (dtype: DTypeGpu) => new CeilOperation(gpuConstructor, dtype)
);
const defaultFloorD = new Dispatcher(
  (dtype: DTypeGpu) => new FloorOperation(gpuConstructor, dtype)
);
const defaultRoundD = new Dispatcher(
  (dtype: DTypeGpu) => new RoundOperation(gpuConstructor, dtype)
);
const defaultClipD = new Dispatcher(
  (dtype: DTypeGpu) => new ClipOperation(gpuConstructor, dtype)
);
const defaultClipBackwardD = new Dispatcher(
  (dtype: DTypeGpu) => new ClipBackwardOperation(gpuConstructor, dtype)
);
const defaultSqrtD = new Dispatcher(
  (dtype: DTypeGpu) => new SqrtOperation(gpuConstructor, dtype)
);
const defaultLogD = new Dispatcher(
  (dtype: DTypeGpu) => new LogOperation(gpuConstructor, dtype)
);
const defaultNegateD = new Dispatcher(
  (dtype: DTypeGpu) => new NegateOperation(gpuConstructor, dtype)
);
const defaultAddMultiplyScalarD = new Dispatcher(
  (dtype: DTypeGpu) => new AddMultiplyScalarOperation(gpuConstructor, dtype)
);
const defaultPowerScalarD = new Dispatcher(
  (dtype: DTypeGpu) => new PowerScalarOperation(gpuConstructor, dtype)
);
const defaultSignD = new Dispatcher(
  (dtype: DTypeGpu) => new SignOperation(gpuConstructor, dtype)
);

//Convolutions
const defaultConvD = new Dispatcher(
  (dtype: DTypeGpu) => new ConvOperation(gpuConstructor, dtype)
);
const defaultAveragePoolD = new Dispatcher(
  (dtype: DTypeGpu) => new AveragePoolOperation(gpuConstructor, dtype)
);
const defaultConvBiasD = new Dispatcher(
  (dtype: DTypeGpu) => new ConvBiasOperation(gpuConstructor, dtype)
);
const defaultConvTransposeD = new Dispatcher(
  (dtype: DTypeGpu) => new ConvTransposeOperation(gpuConstructor, dtype)
);
const defaultPadD = new Dispatcher(
  (dtype: DTypeGpu) => new PadOperation(gpuConstructor, dtype)
);
const defaultUpsampleD = new Dispatcher(
  (dtype: DTypeGpu) => new UpsampleOperation(gpuConstructor, dtype)
);

//Binary operations
const defaultAddD = new Dispatcher(
  (dtype: DTypeGpu) => new AddOperation(gpuConstructor, dtype)
);
const defaultSubtractD = new Dispatcher(
  (dtype: DTypeGpu) => new SubtractOperation(gpuConstructor, dtype)
);
const defaultMultiplyD = new Dispatcher(
  (dtype: DTypeGpu) => new MultiplyOperation(gpuConstructor, dtype)
);
const defaultDivideD = new Dispatcher(
  (dtype: DTypeGpu) => new DivideOperation(gpuConstructor, dtype)
);
const defaultPowerD = new Dispatcher(
  (dtype: DTypeGpu) => new PowerOperation(gpuConstructor, dtype)
);

//Reductions
const defaultMeanD = new Dispatcher(
  (dtype: DTypeGpu) => new ReduceMeanOperation(gpuConstructor, dtype)
);
const defaultMeanSquareD = new Dispatcher(
  (dtype: DTypeGpu) => new ReduceMeanSquareOperation(gpuConstructor, dtype)
);
const defaultSumSquareD = new Dispatcher(
  (dtype: DTypeGpu) => new SumSquareOperation(gpuConstructor, dtype)
);
const defaultSumD = new Dispatcher(
  (dtype: DTypeGpu) => new SumOperation(gpuConstructor, dtype)
);
const defaultProductD = new Dispatcher(
  (dtype: DTypeGpu) => new ProductOperation(gpuConstructor, dtype)
);
const defaultMaxD = new Dispatcher(
  (dtype: DTypeGpu) => new MaxOperation(gpuConstructor, dtype)
);
const defaultArgMaxD = new Dispatcher(
  (dtype: DTypeGpu) => new ArgMaxOperation(gpuConstructor, dtype)
);
const defaultArgMinD = new Dispatcher(
  (dtype: DTypeGpu) => new ArgMinOperation(gpuConstructor, dtype)
);
const defaultMinD = new Dispatcher(
  (dtype: DTypeGpu) => new MinOperation(gpuConstructor, dtype)
);
const defaultLogSumD = new Dispatcher(
  (dtype: DTypeGpu) => new ReduceLogSumOperation(gpuConstructor, dtype)
);
const defaultLogSumExpD = new Dispatcher(
  (dtype: DTypeGpu) => new ReduceLogSumExpOperation(gpuConstructor, dtype)
);

//Util
const defaultConcatD = new Dispatcher(
  (dtype: DTypeGpu) => new ConcatOperation(gpuConstructor, dtype)
);
const defaultSetValuesD = new Dispatcher(
  (dtype: DTypeGpu) => new SetValuesOperation(gpuConstructor, dtype)
);
const defaultCopyD = new Dispatcher(
  (dtype: DTypeGpu) => new CopyOperation(gpuConstructor, dtype)
);
const defaultExpandD = new Dispatcher(
  (dtype: DTypeGpu) => new ExpandOperation(gpuConstructor, dtype)
);
const defaultGatherD = new Dispatcher(
  (dtype: DTypeGpu) => new GatherOperation(gpuConstructor, dtype)
);
const defaultTransposeD = new Dispatcher(
  (dtype: DTypeGpu) => new TransposeOperation(gpuConstructor, dtype)
);
const defaultRepeatD = new Dispatcher(
  (dtype: DTypeGpu) => new RepeatOperation(gpuConstructor, dtype)
);
const defaultSliceD = new Dispatcher(
  (dtype: DTypeGpu) => new SliceOperation(gpuConstructor, dtype)
);
const defaultNormalizeD = new Dispatcher(
  (dtype: DTypeGpu) => new NormalizeOperation(gpuConstructor, dtype)
);
