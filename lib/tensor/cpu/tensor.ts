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
import Tensor, {Activation, PadMode, TensorValues} from '../../types';
import {
  compareShapes,
  computeStrides,
  getSize,
  indexToPos,
} from '../../util/shape';

export class CPUTensor extends Tensor {
  static range(start: number, limit: number, delta: number) {
    const size = Math.max(Math.ceil((limit - start) / delta), 0);
    const values = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      values[i] = start + i * delta;
    }
    return new CPUTensor([size], values);
  }

  public values: TensorValues;

  public shape: ReadonlyArray<number>;

  public strides: ReadonlyArray<number>;

  public size: number;

  public type: string;

  public deleted = false;

  constructor(
    shape: ReadonlyArray<number>,
    values?: TensorValues | number[],
    type?: string
  ) {
    super();

    this.shape = shape;
    this.strides = computeStrides(shape);
    this.size = getSize(shape);

    if (values !== undefined) {
      if (values instanceof Float32Array || values instanceof Int32Array) {
        this.values = values;
        this.type = values instanceof Float32Array ? 'float' : 'int';
      } else if (type === 'int') {
        this.values = Int32Array.from(values);
        this.type = 'int';
      } else {
        this.values = Float32Array.from(values);
        this.type = 'float';
      }
    } else {
      if (type === 'int') {
        this.values = new Int32Array(this.size);
        this.type = 'int';
      } else {
        this.values = new Float32Array(this.size);
        this.type = 'float';
      }
    }
    this.type = 'float';
  }

  getValues() {
    return Promise.resolve(this.values);
  }

  getShape() {
    return this.shape;
  }

  constantLike(value: number): Tensor {
    return new CPUTensor(this.shape, new Float32Array(this.size).fill(value));
  }

  singleConstant(value: number): Tensor {
    return new CPUTensor([1], [value]);
  }

  async cpu(): Promise<CPUTensor> {
    return this;
  }

  delete(): void {
    //@ts-ignore
    this.values = undefined;
    this.deleted = true;
  }

  copy(newShape?: number[]): Tensor {
    if (newShape === undefined) {
      newShape = [...this.shape];
    }
    const values = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) {
      values[i] = this.values[i];
    }
    return new CPUTensor(newShape, values);
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

  setValues(values: Tensor, starts: number[]): Tensor {
    if (!(values instanceof CPUTensor)) {
      throw new Error('Can only set CPU values to CPU values');
    }
    return setValues(this, values, starts);
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

  sin(): Tensor {
    return sin(this);
  }

  cos(): Tensor {
    return cos(this);
  }

  tan(): Tensor {
    return tan(this);
  }

  asin(): Tensor {
    return asin(this);
  }

  acos(): Tensor {
    return acos(this);
  }

  atan(): Tensor {
    return atan(this);
  }

  sinh(): Tensor {
    return sinh(this);
  }

  cosh(): Tensor {
    return cosh(this);
  }

  tanh(): Tensor {
    return tanh(this);
  }

  asinh(): Tensor {
    return asinh(this);
  }

  acosh(): Tensor {
    return acosh(this);
  }

  atanh(): Tensor {
    return atanh(this);
  }

  floor(): Tensor {
    return floor(this);
  }

  ceil(): Tensor {
    return ceil(this);
  }

  round(): Tensor {
    return round(this);
  }

  negate(): Tensor {
    return negate(this);
  }

  powerScalar(power: number, factor: number): Tensor {
    return powerScalar(this, power, factor);
  }

  multiplyScalar(value: number): Tensor {
    return addMultiplyScalar(this, value, 0);
  }

  addScalar(value: number): Tensor {
    return addMultiplyScalar(this, 1, value);
  }

  addMultiplyScalar(factor: number, add: number): Tensor {
    return addMultiplyScalar(this, factor, add);
  }

  sign(): Tensor {
    return sign(this);
  }

  clip(min?: number, max?: number): Tensor {
    return clip(this, min, max);
  }

  clipBackward(grad: Tensor, min?: number, max?: number): Tensor {
    if (!(grad instanceof CPUTensor)) {
      throw new Error('Can only do clipBackward with CPUTensor');
    }
    return clipBackward(this, grad, this.getShape(), min, max);
  }

  sigmoid(): Tensor {
    return sigmoid(this);
  }

  hardSigmoid(alpha: number, beta: number): Tensor {
    return hardSigmoid(this, alpha, beta);
  }

  add_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }
    return add(th, tensor, resultShape, alpha, beta);
  }

  subtract_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only subtract CPU tensor to CPU tensor');
    }
    return subtract(th, tensor, resultShape, alpha, beta);
  }

  multiply_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[],
    alpha: number
  ): Tensor {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }
    return multiply(th, tensor, resultShape, alpha);
  }

  divide_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[],
    alpha: number
  ): Tensor {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }
    return divide(th, tensor, resultShape, alpha);
  }

  power_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[]
  ): Tensor {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only take CPU tensor to power of CPU tensor');
    }
    return power(th, tensor, resultShape);
  }

  matMul(tensor: Tensor): Tensor {
    if (!(tensor instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }

    return matMul(this, tensor);
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
      !(b instanceof CPUTensor && (c === undefined || c instanceof CPUTensor))
    ) {
      throw new Error('Can only do gemm with CPU tensors');
    }
    return gemm(this, b, aTranspose, bTranspose, alpha, beta, c as CPUTensor);
  }

  sum_impl(axes: number[], keepDims: boolean): Tensor {
    return sum(this, axes, keepDims);
  }

  sumSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return sumSquare(this, axes, keepDims);
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

  reduceMean_impl(axes: number[], keepDims: boolean): Tensor {
    return reduceMean(this, axes, keepDims);
  }

  reduceMeanSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return reduceMeanSquare(this, axes, keepDims);
  }

  reduceLogSum_impl(axes: number[], keepDims: boolean): Tensor {
    return reduceLogSum(this, axes, keepDims);
  }

  reduceLogSumExp_impl(axes: number[], keepDims: boolean): Tensor {
    return reduceLogSumExp(this, axes, keepDims);
  }

  conv_impl(
    kernel: Tensor,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[],
    activation: Activation,
    bias?: Tensor
  ): Tensor {
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
      bias as CPUTensor
    );
  }

  protected convTranspose_impl(
    kernel: Tensor,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[]
  ): Tensor {
    if (!(kernel instanceof CPUTensor)) {
      throw new Error(
        'Can only do transpose convolution of CPU tensor with CPU tensor'
      );
    }

    return convTranspose(this, kernel, dilations, group, pads, strides);
  }

  pad_impl(pads: number[], mode: PadMode, value: number): Tensor {
    return pad(this, pads, mode, value);
  }

  averagePool_impl(
    kernelShape: number[],
    pads: number[],
    strides: number[],
    includePad: boolean
  ): Tensor {
    return averagePool(this, kernelShape, pads, strides, includePad);
  }

  reshape_impl(shape: number[], copy: boolean): Tensor {
    if (copy) {
      return this.copy(shape);
    } else {
      return new CPUTensor(shape, this.values, this.type);
    }
  }

  concat(tensor: Tensor, axis: number): Tensor {
    if (!(tensor instanceof CPUTensor)) {
      throw new Error('Can only concat CPU tensor to CPU tensor');
    }
    if (axis < 0) {
      axis += this.shape.length;
    }
    return concat(this, tensor, axis);
  }

  transpose_impl(permutation: number[]): Tensor {
    return transpose(this, permutation);
  }

  repeat(repeats: number[]): Tensor {
    return repeat(this, repeats);
  }

  expand(shape: readonly number[]): Tensor {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [_shape, goal, resultShape] = this.alignShapes(this.shape, shape);
    if (compareShapes(this.shape, resultShape)) {
      return this.copy();
    }
    return expand(this.reshape(_shape, false) as CPUTensor, resultShape);
  }

  gather(axis: number, indices: CPUTensor): Tensor {
    return gather(this, axis, indices);
  }

  slice_impl(
    starts: number[],
    ends: number[],
    axes: number[],
    steps: number[]
  ): Tensor {
    return slice(this, starts, ends, axes, steps);
  }

  upsample(scales: number[]): Tensor {
    return upsample(this, scales);
  }

  normalize(
    mean: Tensor,
    variance: Tensor,
    epsilon: number,
    scale: Tensor,
    bias: Tensor
  ): Tensor {
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
