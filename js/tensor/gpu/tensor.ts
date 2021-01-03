import Tensor from '../../types';

import { getSize } from '../../util/shape';

import REGL, { Framebuffer2D } from 'regl';
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

let glContext = document.createElement("canvas").getContext("webgl");
export let gl = REGL({
  gl: glContext,
  extensions: ['OES_texture_float']
});

export default class GPUTensor extends Tensor {
  public framebuffer: Framebuffer2D;

  public size: number;
  public textureSize: number;
  public arraySize: number;

  public shape: readonly number[];

  constructor(values: Float32Array | Framebuffer2D, shape: readonly number[]) {
    super();

    this.size = getSize(shape);
    this.shape = shape;

    this.textureSize = Math.ceil(this.size / 4)
    this.arraySize = this.textureSize*4;

    if (values instanceof Float32Array) {
      const vals = new Float32Array(this.arraySize);
      for (let i = 0; i < this.size; i++) {
        vals[i] = values[i];
      }
      for (let i = this.size; i < this.arraySize; i++) {
        vals[i] = 0;
      }

      // TODO: Actually use the height of the texture
      // Large tensors can otherwise not be stored effectively

      const texture = gl.texture({
        width: this.textureSize,
        height: 1,
        format: 'rgba',
        type: 'float',
        data: vals,
      });

      this.framebuffer = gl.framebuffer({
        color: texture,
        width: this.textureSize,
        height: 1,
        depthStencil: false
      });
    } else {
      this.framebuffer = values;
    }
  }

  getValues(): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      gl({framebuffer: this.framebuffer})(() => {
        let result = new Float32Array(this.arraySize);
        result = gl.read(result);
        resolve(result.subarray(0, this.size));
      });
    });
  }

  getShape(): readonly number[] {
    return this.shape;
  }

  delete(): void {
    this.framebuffer.destroy();
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
      throw new Error('Can only add GPU tensor to GPU tensor');
    }
    return subtract(this, tensor);
  }

  multiply(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only add GPU tensor to GPU tensor');
    }
    return multiply(this, tensor);
  }

  divide(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only add GPU tensor to GPU tensor');
    }
    return divide(this, tensor);
  }

  matMul(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only add GPU tensor to GPU tensor');
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
