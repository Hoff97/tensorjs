import {argMax} from '../../ops/cpu/argMax';
import {averagePool} from '../../ops/cpu/averagePool';
import {
  abs,
  acos,
  acosh,
  add,
  addMultiplyScalar,
  asin,
  asinh,
  atan,
  atanh,
  ceil,
  clip,
  clipBackward,
  cos,
  cosh,
  divide,
  exp,
  floor,
  hardSigmoid,
  log,
  multiply,
  negate,
  power,
  powerScalar,
  round,
  sigmoid,
  sign,
  sin,
  sinh,
  sqrt,
  subtract,
  tan,
  tanh,
} from '../../ops/cpu/basic';
import {concat} from '../../ops/cpu/concat';
import {conv} from '../../ops/cpu/conv';
import {convTranspose} from '../../ops/cpu/convTranspose';
import {expand} from '../../ops/cpu/expand';
import {gather} from '../../ops/cpu/gather';
import {gemm} from '../../ops/cpu/gemm';
import {matMul} from '../../ops/cpu/matMul';
import {max} from '../../ops/cpu/max';
import {min} from '../../ops/cpu/min';
import {normalize} from '../../ops/cpu/normalize';
import {pad} from '../../ops/cpu/pad';
import {product} from '../../ops/cpu/product';
import {reduceLogSum} from '../../ops/cpu/reduceLogSum';
import {reduceLogSumExp} from '../../ops/cpu/reduceLogSumExp';
import {reduceMean} from '../../ops/cpu/reduceMean';
import {reduceMeanSquare} from '../../ops/cpu/reduceMeanSquare';
import {repeat} from '../../ops/cpu/repeat';
import {setValues} from '../../ops/cpu/setValues';
import {slice} from '../../ops/cpu/slice';
import {sum} from '../../ops/cpu/sum';
import {sumSquare} from '../../ops/cpu/sumSquare';
import {transpose} from '../../ops/cpu/transpose';
import {upsample} from '../../ops/cpu/upsample';
import Tensor, {
  Activation,
  DType,
  PadMode,
  TensorValues,
  tensorValuesConstructor,
} from '../../types';
import {
  compareShapes,
  computeStrides,
  getSize,
  indexToPos,
} from '../../util/shape';

export class CPUTensor<DTpe extends DType = 'float32'> extends Tensor<DTpe> {
  static range(start: number, limit: number, delta: number) {
    const size = Math.max(Math.ceil((limit - start) / delta), 0);
    const values = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      values[i] = start + i * delta;
    }
    return new CPUTensor([size], values);
  }

  /**
   * Array of values of the tensor in contiguous layout
   */
  public values: TensorValues[DTpe];

  /**
   * Shape of the tensor
   */
  public shape: ReadonlyArray<number>;

  /**
   * Strides for all dimensions, ie. the step size per dimension in the contiguous layout
   */
  public strides: ReadonlyArray<number>;

  /**
   * Total number of entries in the tensor
   */
  public size: number;

  /**
   * If this tensor was already deleted
   */
  public deleted = false;

  constructor(
    shape: ReadonlyArray<number>,
    values?: TensorValues[DTpe] | number[],
    dtype?: DTpe
  ) {
    super(dtype || ('float32' as any));

    this.shape = shape;
    this.strides = computeStrides(shape);
    this.size = getSize(shape);

    if (values !== undefined) {
      if (values instanceof Array) {
        this.values = new tensorValuesConstructor[this.dtype](
          values
        ) as TensorValues[DTpe];
      } else {
        this.values = values;
      }
    } else {
      this.values = new tensorValuesConstructor[this.dtype](
        this.size
      ) as TensorValues[DTpe];
    }
  }

  getValues() {
    return Promise.resolve(this.values);
  }

  getShape() {
    return this.shape;
  }

  constantLike(value: number): Tensor<DTpe> {
    return new CPUTensor<DTpe>(
      this.shape,
      new tensorValuesConstructor[this.dtype](this.size).fill(
        value
      ) as TensorValues[DTpe],
      this.dtype
    );
  }

  singleConstant(value: number): Tensor<DTpe> {
    return new CPUTensor([1], [value], this.dtype);
  }

  cast<DTpe2 extends DType>(dtype: DTpe2): Tensor<DTpe2> {
    // TODO: Catch special cases here
    // Eg casting to the same type
    return new CPUTensor(this.shape, Array.from(this.values), dtype);
  }

  delete(): void {
    //@ts-ignore
    this.values = undefined;
    this.deleted = true;
  }

  copy(newShape?: number[]): Tensor<DTpe> {
    if (newShape === undefined) {
      newShape = [...this.shape];
    }
    const values = new tensorValuesConstructor[this.dtype](
      this.size
    ) as TensorValues[DTpe];
    for (let i = 0; i < this.size; i++) {
      values[i] = this.values[i];
    }
    return new CPUTensor<DTpe>(newShape, values, this.dtype);
  }

  get(index: number[] | number): number {
    let pos: number;
    if (Array.isArray(index)) {
      pos = indexToPos(index, this.strides, this.shape);
    } else {
      pos = index;
    }

    return this.values[pos];
  }

  set(index: number[] | number, value: number) {
    let pos: number;
    if (Array.isArray(index)) {
      pos = indexToPos(index, this.strides);
    } else {
      pos = index;
    }

    this.values[pos] = value;
  }

  setValues(values: Tensor<DTpe>, starts: number[]): Tensor<DTpe> {
    if (!(values instanceof CPUTensor)) {
      throw new Error('Can only set CPU values to CPU values');
    }
    return setValues(this, values, starts);
  }

  exp(): Tensor<DTpe> {
    return exp(this);
  }

  log(): Tensor<DTpe> {
    return log(this);
  }

  sqrt(): Tensor<DTpe> {
    return sqrt(this);
  }

  abs(): Tensor<DTpe> {
    return abs(this);
  }

  sin(): Tensor<DTpe> {
    return sin(this);
  }

  cos(): Tensor<DTpe> {
    return cos(this);
  }

  tan(): Tensor<DTpe> {
    return tan(this);
  }

  asin(): Tensor<DTpe> {
    return asin(this);
  }

  acos(): Tensor<DTpe> {
    return acos(this);
  }

  atan(): Tensor<DTpe> {
    return atan(this);
  }

  sinh(): Tensor<DTpe> {
    return sinh(this);
  }

  cosh(): Tensor<DTpe> {
    return cosh(this);
  }

  tanh(): Tensor<DTpe> {
    return tanh(this);
  }

  asinh(): Tensor<DTpe> {
    return asinh(this);
  }

  acosh(): Tensor<DTpe> {
    return acosh(this);
  }

  atanh(): Tensor<DTpe> {
    return atanh(this);
  }

  floor(): Tensor<DTpe> {
    return floor(this);
  }

  ceil(): Tensor<DTpe> {
    return ceil(this);
  }

  round(): Tensor<DTpe> {
    return round(this);
  }

  negate(): Tensor<DTpe> {
    return negate(this);
  }

  powerScalar(power: number, factor: number): Tensor<DTpe> {
    return powerScalar(this, power, factor);
  }

  multiplyScalar(value: number): Tensor<DTpe> {
    return addMultiplyScalar(this, value, 0);
  }

  addScalar(value: number): Tensor<DTpe> {
    return addMultiplyScalar(this, 1, value);
  }

  addMultiplyScalar(factor: number, add: number): Tensor<DTpe> {
    return addMultiplyScalar(this, factor, add);
  }

  sign(): Tensor<DTpe> {
    return sign(this);
  }

  clip(min?: number, max?: number): Tensor<DTpe> {
    return clip(this, min, max);
  }

  clipBackward(grad: Tensor<DTpe>, min?: number, max?: number): Tensor<DTpe> {
    if (!(grad instanceof CPUTensor)) {
      throw new Error('Can only do clipBackward with CPUTensor');
    }
    return clipBackward(this, grad, this.getShape(), min, max);
  }

  sigmoid(): Tensor<DTpe> {
    return sigmoid(this);
  }

  hardSigmoid(alpha: number, beta: number): Tensor<DTpe> {
    return hardSigmoid(this, alpha, beta);
  }

  add_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }
    return add(th, tensor, resultShape, alpha, beta);
  }

  subtract_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only subtract CPU tensor to CPU tensor');
    }
    return subtract(th, tensor, resultShape, alpha, beta);
  }

  multiply_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }
    return multiply(th, tensor, resultShape, alpha);
  }

  divide_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }
    return divide(th, tensor, resultShape, alpha);
  }

  power_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[]
  ): Tensor<DTpe> {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only take CPU tensor to power of CPU tensor');
    }
    return power(th, tensor, resultShape);
  }

  matMul(tensor: Tensor<DTpe>): Tensor<DTpe> {
    if (!(tensor instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }

    return matMul(this, tensor);
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
      !(b instanceof CPUTensor && (c === undefined || c instanceof CPUTensor))
    ) {
      throw new Error('Can only do gemm with CPU tensors');
    }
    return gemm(
      this,
      b,
      aTranspose,
      bTranspose,
      alpha,
      beta,
      c as CPUTensor<DTpe>
    );
  }

  sum_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return sum(this, axes, keepDims);
  }

  sumSquare_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return sumSquare(this, axes, keepDims);
  }

  product_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return product(this, axes, keepDims);
  }

  max_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return max(this, axes, keepDims);
  }

  min_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return min(this, axes, keepDims);
  }

  protected argMax_impl(axes: number[], selectLast: boolean): Tensor<'uint32'> {
    return argMax(this, axes, selectLast);
  }

  reduceMean_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return reduceMean(this, axes, keepDims);
  }

  reduceMeanSquare_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return reduceMeanSquare(this, axes, keepDims);
  }

  reduceLogSum_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return reduceLogSum(this, axes, keepDims);
  }

  reduceLogSumExp_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return reduceLogSumExp(this, axes, keepDims);
  }

  conv_impl(
    kernel: Tensor<DTpe>,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[],
    activation: Activation,
    bias?: Tensor<DTpe>
  ): Tensor<DTpe> {
    if (
      !(kernel instanceof CPUTensor) ||
      (bias !== undefined && !(bias instanceof CPUTensor))
    ) {
      throw new Error('Can only do convolution of CPU tensor with CPU tensor');
    }
    return conv(
      this,
      kernel,
      dilations,
      group,
      pads,
      strides,
      activation,
      bias as CPUTensor<DTpe>
    );
  }

  protected convTranspose_impl(
    kernel: Tensor<DTpe>,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[]
  ): Tensor<DTpe> {
    if (!(kernel instanceof CPUTensor)) {
      throw new Error(
        'Can only do transpose convolution of CPU tensor with CPU tensor'
      );
    }

    return convTranspose(this, kernel, dilations, group, pads, strides);
  }

  pad_impl(pads: number[], mode: PadMode, value: number): Tensor<DTpe> {
    return pad(this, pads, mode, value);
  }

  averagePool_impl(
    kernelShape: number[],
    pads: number[],
    strides: number[],
    includePad: boolean
  ): Tensor<DTpe> {
    return averagePool(this, kernelShape, pads, strides, includePad);
  }

  reshape_impl(shape: number[], copy: boolean): Tensor<DTpe> {
    if (copy) {
      return this.copy(shape);
    } else {
      return new CPUTensor(shape, this.values, this.dtype);
    }
  }

  concat(tensor: Tensor<DTpe>, axis: number): Tensor<DTpe> {
    if (!(tensor instanceof CPUTensor)) {
      throw new Error('Can only concat CPU tensor to CPU tensor');
    }
    if (axis < 0) {
      axis += this.shape.length;
    }
    return concat(this, tensor, axis);
  }

  transpose_impl(permutation: number[]): Tensor<DTpe> {
    return transpose(this, permutation);
  }

  repeat(repeats: number[]): Tensor<DTpe> {
    return repeat(this, repeats);
  }

  expand(shape: readonly number[]): Tensor<DTpe> {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [_shape, goal, resultShape] = this.alignShapes(this.shape, shape);
    if (compareShapes(this.shape, resultShape)) {
      return this.copy();
    }
    return expand(this.reshape(_shape, false) as CPUTensor<DTpe>, resultShape);
  }

  gather(axis: number, indices: CPUTensor<'uint32'>): Tensor<DTpe> {
    return gather(this, axis, indices);
  }

  slice_impl(
    starts: number[],
    ends: number[],
    axes: number[],
    steps: number[]
  ): Tensor<DTpe> {
    return slice(this, starts, ends, axes, steps);
  }

  upsample(scales: number[]): Tensor<DTpe> {
    return upsample(this, scales);
  }

  normalize(
    mean: Tensor<DTpe>,
    variance: Tensor<DTpe>,
    epsilon: number,
    scale: Tensor<DTpe>,
    bias: Tensor<DTpe>
  ): Tensor<DTpe> {
    if (
      !(mean instanceof CPUTensor) ||
      !(variance instanceof CPUTensor) ||
      !(scale instanceof CPUTensor) ||
      !(bias instanceof CPUTensor)
    ) {
      throw new Error('Can only normalize with CPU tensors');
    }
    return normalize(this, mean, variance, epsilon, scale, bias);
  }
}
