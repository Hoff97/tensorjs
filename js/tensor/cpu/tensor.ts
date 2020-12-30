import {
  add, divide, exp, log, matMul, multiply, sqrt, subtract
} from '../../ops/cpu/basic';
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
    return this.values;
  }

  getShape() {
    return this.shape;
  }

  get(index: number[] | number): number {
    let pos: number;
    if (Array.isArray(index)) {
      pos = indexToPos(index, this.strides);
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

  add(tensor: Tensor): Tensor {
    if (!(tensor instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }
    return add(this, tensor);
  }

  subtract(tensor: Tensor): Tensor {
    if (!(tensor instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }
    return subtract(this, tensor);
  }

  multiply(tensor: Tensor): Tensor {
    if (!(tensor instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }

    return multiply(this, tensor);
  }

  divide(tensor: Tensor): Tensor {
    if (!(tensor instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }

    return divide(this, tensor);
  }

  matMul(tensor: Tensor): Tensor {
    if (!(tensor instanceof CPUTensor)) {
      throw new Error('Can only add CPU tensor to CPU tensor');
    }

    return matMul(this, tensor);
  }
}
