import {
  add, divide, exp, log, multiply, sqrt, subtract
} from '../../ops/cpu/basic';
import { conv } from '../../ops/cpu/conv';
import { matMul } from '../../ops/cpu/matMul';
import { max } from '../../ops/cpu/max';
import { min } from '../../ops/cpu/min';
import { product } from '../../ops/cpu/product';
import { sum } from '../../ops/cpu/sum';
import Tensor from '../../types';
import { computeStrides, getSize, indexToPos } from '../../util/shape';

export default class CPUTensor extends Tensor {
  private values: Float32Array;

  public shape: ReadonlyArray<number>;

  public strides: ReadonlyArray<number>;

  public size: number;

  constructor(shape: ReadonlyArray<number>, values?: Float32Array | number[]) {
    super();

    this.shape = shape;
    this.strides = computeStrides(shape);
    this.size = getSize(shape);

    if (values !== undefined) {
      if (values instanceof Float32Array) {
        this.values = values;
      } else {
        this.values = Float32Array.from(values);
      }
    } else {
      this.values = new Float32Array(this.size);
    }
  }

  getValues() {
    return Promise.resolve(this.values);
  }

  getShape() {
    return this.shape;
  }

  delete(): void {
    // TODO: Maybe set values to empty array?
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

  add_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof CPUTensor) || !(th instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }
    console.log(th, tensor, resultShape);
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

  matMul(tensor: Tensor): Tensor {
    if (!(tensor instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }

    return matMul(this, tensor);
  }

  sum_impl(axes: number[]): Tensor {
    return sum(this, axes);
  }

  product_impl(axes: number[]): Tensor {
    return product(this, axes);
  }

  max_impl(axes: number[]): Tensor {
    return max(this, axes);
  }

  min_impl(axes: number[]): Tensor {
    return min(this, axes);
  }

  conv_impl(kernel: Tensor, dilations: number[], group: number, pads: number[], strides: number[], bias?: Tensor): Tensor {
    if (!(kernel instanceof CPUTensor) || (bias !== undefined && !(bias instanceof CPUTensor))) {
      throw new Error('Can only do convolution of CPU tensor with CPU tensor');
    }
    return conv(this, kernel, dilations, group, pads, strides, bias as CPUTensor);
  }

  reshape(shape: number[]): Tensor {
    return new CPUTensor(shape, this.values);
  }
}
