import { averagePool } from '../../ops/cpu/averagePool';
import {
  abs,
  add, ceil, clip, divide, exp, floor, log, multiply, power, sqrt, subtract
} from '../../ops/cpu/basic';
import { concat } from '../../ops/cpu/concat';
import { conv } from '../../ops/cpu/conv';
import { expand } from '../../ops/cpu/expand';
import { gather } from '../../ops/cpu/gather';
import { gemm } from '../../ops/cpu/gemm';
import { matMul } from '../../ops/cpu/matMul';
import { max } from '../../ops/cpu/max';
import { min } from '../../ops/cpu/min';
import { pad } from '../../ops/cpu/pad';
import { product } from '../../ops/cpu/product';
import { reduceMean } from '../../ops/cpu/reduceMean';
import { reduceMeanSquare } from '../../ops/cpu/reduceMeanSquare';
import { repeat } from '../../ops/cpu/repeat';
import { sum } from '../../ops/cpu/sum';
import { sumSquare } from '../../ops/cpu/sumSquare';
import { transpose } from '../../ops/cpu/transpose';
import Tensor, { PadMode, TensorValues } from '../../types';
import { compareShapes, computeStrides, getSize, indexToPos } from '../../util/shape';

export class CPUTensor extends Tensor {
  public values: TensorValues;

  public shape: ReadonlyArray<number>;

  public strides: ReadonlyArray<number>;

  public size: number;

  public type: string;

  public deleted: boolean = false;

  constructor(shape: ReadonlyArray<number>, values?: TensorValues | number[], type?: string) {
    super();

    this.shape = shape;
    this.strides = computeStrides(shape);
    this.size = getSize(shape);

    if (values !== undefined) {
      if (values instanceof Float32Array || values instanceof Int32Array) {
        this.values = values;
        this.type = values instanceof Float32Array ? "float" : "int";
      } else if (type === "int") {
        this.values = Int32Array.from(values);
        this.type = "int";
      } else {
        this.values = Float32Array.from(values);
        this.type = "float";
      }
    } else {
      if (type === "int") {
        this.values = new Int32Array(this.size);
        this.type = "int";
      } else {
        this.values = new Float32Array(this.size);
        this.type = "float";
      }
    }
  }

  getValues() {
    return Promise.resolve(this.values);
  }

  getShape() {
    return this.shape;
  }

  async cpu(): Promise<CPUTensor> {
    return this;
  }

  delete(): void {
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

  floor(): Tensor {
    return floor(this);
  }

  ceil(): Tensor {
    return ceil(this);
  }

  clip(min?: number, max?: number): Tensor {
    return clip(this, min, max);
  }

  add_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }
    return add(th, tensor, resultShape);
  }

  subtract_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only subtract CPU tensor to CPU tensor');
    }
    return subtract(th, tensor, resultShape);
  }

  multiply_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }
    return multiply(th, tensor, resultShape);
  }

  divide_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }
    return divide(th, tensor, resultShape);
  }

  power_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
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

  gemm_impl(b: Tensor, aTranspose: boolean, bTranspose: boolean, alpha: number, beta: number, c?: Tensor): Tensor {
    if (!(b instanceof CPUTensor && (c === undefined || c instanceof CPUTensor))) {
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

  conv_impl(kernel: Tensor, dilations: number[], group: number, pads: number[], strides: number[], bias?: Tensor): Tensor {
    if (!(kernel instanceof CPUTensor) || (bias !== undefined && !(bias instanceof CPUTensor))) {
      throw new Error('Can only do convolution of CPU tensor with CPU tensor');
    }
    return conv(this, kernel, dilations, group, pads, strides, bias as CPUTensor);
  }

  pad_impl(pads: number[], mode: PadMode, value: number): Tensor {
    return pad(this, pads, mode, value);
  }

  averagePool_impl(kernelShape: number[],
                    pads: number[],
                    strides: number[],
                    includePad: boolean): Tensor {
    return averagePool(this, kernelShape, pads, strides, includePad);
  }

  reshape_impl(shape: number[]): Tensor {
    return this.copy(shape);
  }

  concat(tensor: Tensor, axis: number): Tensor {
    if (!(tensor instanceof CPUTensor)) {
      throw new Error('Can only concat CPU tensor to CPU tensor');
    }
    return concat(this, tensor, axis);
  }

  transpose_impl(permutation: number[]): Tensor {
    return transpose(this, permutation);
  }

  repeat(repeats: number[]): Tensor {
    return repeat(this, repeats);
  }

  expand(shape: number[]): Tensor {
    const [_shape, goal, resultShape] = this.alignShapes(this.shape, shape);
    if (compareShapes(this.shape, resultShape)) {
      return this.copy();
    }
    return expand(this.reshape(_shape) as CPUTensor, resultShape);
  }

  gather(axis: number, indices: CPUTensor): Tensor {
    return gather(this, axis, indices);
  }
}
