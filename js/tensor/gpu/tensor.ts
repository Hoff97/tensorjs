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

let glContext = document.createElement("canvas").getContext("webgl", {
  preserveDrawingBuffer: true,
  failIfMajorPerformanceCaveat: true
});
export let gl = REGL({
  gl: glContext,
  extensions: ['OES_texture_float']
});

let tensorID = 0;

export default class GPUTensor extends Tensor {
  public framebuffer: Framebuffer2D;

  public size: number;
  public textureSize: number;
  public arraySize: number;

  public shape: readonly number[];

  public id: number;
  public deleted: boolean = false;

  constructor(values: Float32Array | Framebuffer2D, shape: readonly number[]) {
    super();

    this.size = getSize(shape);
    this.shape = shape;

    this.textureSize = Math.ceil(this.size / 4)
    this.arraySize = this.textureSize*4;

    this.id = tensorID++;

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
    this.deleted = true;
  }

  private expDest?: GPUTensor;
  exp(): Tensor {
    if (this.expDest && !this.expDest.deleted) {
      return exp(this, this.expDest);
    }
    const result = exp(this);
    this.expDest = result;
    return result;
  }

  private logDest?: GPUTensor;
  log(): Tensor {
    if (this.logDest && !this.logDest.deleted) {
      return log(this, this.logDest);
    }
    const result = log(this);
    this.logDest = result;
    return result;
  }

  private sqrtDest?: GPUTensor;
  sqrt(): Tensor {
    if (this.sqrtDest && !this.sqrtDest.deleted) {
      return sqrt(this, this.sqrtDest);
    }
    const result = sqrt(this);
    this.sqrtDest = result;
    return result;
  }

  private addDest: {[id: number]: GPUTensor} = {};
  add(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only add GPU tensor to GPU tensor');
    }
    if (this.addDest[tensor.id] && !this.addDest[tensor.id].deleted) {
      return add(this, tensor, this.addDest[tensor.id]);
    }
    const result = add(this, tensor);
    this.addDest[tensor.id] = result;
    return result;
  }

  private subtractDest: {[id: number]: GPUTensor} = {};
  subtract(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only add GPU tensor to GPU tensor');
    }
    if (this.subtractDest[tensor.id] && !this.subtractDest[tensor.id].deleted) {
      return subtract(this, tensor, this.subtractDest[tensor.id]);
    }
    const result = subtract(this, tensor);
    this.subtractDest[tensor.id] = result;
    return result;
  }

  private multiplyDest: {[id: number]: GPUTensor} = {};
  multiply(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only add GPU tensor to GPU tensor');
    }
    if (this.multiplyDest[tensor.id] && !this.multiplyDest[tensor.id].deleted) {
      return multiply(this, tensor, this.multiplyDest[tensor.id]);
    }
    const result = multiply(this, tensor);
    this.multiplyDest[tensor.id] = result;
    return result;
  }

  private divideDest: {[id: number]: GPUTensor} = {};
  divide(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only add GPU tensor to GPU tensor');
    }
    if (this.divideDest[tensor.id] && !this.divideDest[tensor.id].deleted) {
      return divide(this, tensor, this.divideDest[tensor.id]);
    }
    const result = divide(this, tensor);
    this.divideDest[tensor.id] = result;
    return result;
  }

  private matMulDest: {[id: number]: GPUTensor} = {};
  matMul(tensor: Tensor): Tensor {
    if (!(tensor instanceof GPUTensor)) {
      throw new Error('Can only add GPU tensor to GPU tensor');
    }
    if (this.matMulDest[tensor.id] && !this.matMulDest[tensor.id].deleted) {
      return matmul(this, tensor, this.matMulDest[tensor.id]);
    }
    const result = matmul(this, tensor);
    this.matMulDest[tensor.id] = result;
    return result;
  }

  private sumDest: {[axes: string]: GPUTensor} = {};
  sum_impl(axes: number[]): Tensor {
    const key = `${axes}`;
    if (this.sumDest[key] && !this.sumDest[key].deleted) {
      return sum(this, axes, this.sumDest[key]);
    }
    const result = sum(this, axes);
    this.sumDest[key] = result;
    return result;
  }

  private productDest: {[axes: string]: GPUTensor} = {};
  product_impl(axes: number[]): Tensor {
    const key = `${axes}`;
    if (this.productDest[key] && !this.productDest[key].deleted) {
      return product(this, axes, this.productDest[key]);
    }
    const result = product(this, axes);
    this.productDest[key] = result;
    return result;
  }

  private maxDest: {[axes: string]: GPUTensor} = {};
  max_impl(axes: number[]): Tensor {
    const key = `${axes}`;
    if (this.maxDest[key] && !this.maxDest[key].deleted) {
      return max(this, axes, this.maxDest[key]);
    }
    const result = max(this, axes);
    this.maxDest[key] = result;
    return result;
  }

  private minDest: {[axes: string]: GPUTensor} = {};
  min_impl(axes: number[]): Tensor {
    const key = `${axes}`;
    if (this.minDest[key] && !this.minDest[key].deleted) {
      return min(this, axes, this.minDest[key]);
    }
    const result = min(this, axes);
    this.minDest[key] = result;
    return result;
  }
}
