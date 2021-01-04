import Tensor from '../../types';

import { getSize } from '../../util/shape';

import { exp } from '../../ops/gpu/exp';
import { log } from '../../ops/gpu/log';
import { sqrt } from '../../ops/gpu/sqrt';
import { add } from '../../ops/gpu/add';
import { subtract } from '../../ops/gpu/subtract';
import { multiply } from '../../ops/gpu/multiply';
import { divide } from '../../ops/gpu/divide';
import { matmul } from '../../ops/gpu/matmul';
import { sum } from '../../ops/gpu/sum';
import { product } from '../../ops/gpu/product';
import { max } from '../../ops/gpu/max';
import { min } from '../../ops/gpu/min';
import { defaultAllocator, gl } from './gl';
import { MemoryEntry } from './memory';


export default class GPUTensor extends Tensor {
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

  delete(): void {
    this.deleted = true;
    defaultAllocator.deallocate(this.memory);
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
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only add GPU tensor to GPU tensor');
    }
    return add(this, tensor);
  }

  subtract(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only subtract GPU tensor to GPU tensor');
    }
    return subtract(this, tensor);
  }

  multiply(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only multiply GPU tensor to GPU tensor');
    }
    return multiply(this, tensor);
  }

  divide(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only divide GPU tensor to GPU tensor');
    }
    return divide(this, tensor);
  }

  matMul(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only matrix multiply GPU tensor to GPU tensor');
    }
    return matmul(this, tensor);
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
}
