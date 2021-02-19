import {Activation} from '../../library';
import Tensor, {PadMode} from '../../types';
import {compareShapes} from '../../util/shape';

import {Tensor as WT} from '../../wasm/rust_wasm_tensor';
import {CPUTensor} from '../cpu/tensor';

let WASMT: typeof WT;
export const wasmLoaded: Promise<void> = new Promise(resolve => {
  import('../../wasm/rust_wasm_tensor').then(x => {
    WASMT = x.Tensor;
    resolve();
  });
});

export class WASMTensor extends Tensor {
  static range(start: number, limit: number, delta: number) {
    const size = Math.max(Math.ceil((limit - start) / delta), 0);
    const values = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      values[i] = start + i * delta;
    }
    return new WASMTensor(values, new Uint32Array([size]));
  }

  public wasmTensor: WT;

  constructor(values: Float32Array | WT, shape?: Uint32Array) {
    super();

    if (values instanceof Float32Array) {
      if (shape === undefined) {
        throw new Error(
          'Need the shape when creating a Wasm tensor from values'
        );
      }
      this.wasmTensor = WASMT.create(shape, values);
    } else {
      this.wasmTensor = values;
    }
  }

  getValues() {
    return Promise.resolve(this.wasmTensor.get_vals());
  }

  getShape(): readonly number[] {
    return Array.from(this.wasmTensor.get_shape());
  }

  constantLike(value: number): Tensor {
    // TODO: Maybe more efficient in WASM?
    return new WASMTensor(
      new Float32Array(this.wasmTensor.size).fill(value),
      this.wasmTensor.get_shape()
    );
  }

  singleConstant(value: number): Tensor {
    return new WASMTensor(new Float32Array([value]), new Uint32Array([1]));
  }

  async wasm(): Promise<WASMTensor> {
    return this;
  }

  delete(): void {
    if (this.wasmTensor !== undefined) {
      this.wasmTensor.free();
      //@ts-ignore
      this.wasmTensor = undefined;
    }
  }

  copy(): Tensor {
    return new WASMTensor(this.wasmTensor.copy());
  }

  exp(): Tensor {
    return new WASMTensor(this.wasmTensor.exp());
  }

  log(): Tensor {
    return new WASMTensor(this.wasmTensor.log());
  }

  sqrt(): Tensor {
    return new WASMTensor(this.wasmTensor.sqrt());
  }

  abs(): Tensor {
    return new WASMTensor(this.wasmTensor.abs());
  }

  sin(): Tensor {
    return new WASMTensor(this.wasmTensor.sin());
  }

  cos(): Tensor {
    return new WASMTensor(this.wasmTensor.cos());
  }

  tan(): Tensor {
    return new WASMTensor(this.wasmTensor.tan());
  }

  asin(): Tensor {
    return new WASMTensor(this.wasmTensor.asin());
  }

  acos(): Tensor {
    return new WASMTensor(this.wasmTensor.acos());
  }

  atan(): Tensor {
    return new WASMTensor(this.wasmTensor.atan());
  }

  sinh(): Tensor {
    return new WASMTensor(this.wasmTensor.sinh());
  }

  cosh(): Tensor {
    return new WASMTensor(this.wasmTensor.cosh());
  }

  tanh(): Tensor {
    return new WASMTensor(this.wasmTensor.tanh());
  }

  asinh(): Tensor {
    return new WASMTensor(this.wasmTensor.asinh());
  }

  acosh(): Tensor {
    return new WASMTensor(this.wasmTensor.acosh());
  }

  atanh(): Tensor {
    return new WASMTensor(this.wasmTensor.atanh());
  }

  sigmoid(): Tensor {
    return new WASMTensor(this.wasmTensor.sigmoid());
  }

  hardSigmoid(alpha: number, beta: number): Tensor {
    return new WASMTensor(this.wasmTensor.hard_sigmoid(alpha, beta));
  }

  negate(): Tensor {
    return new WASMTensor(this.wasmTensor.negate());
  }

  powerScalar(power: number, factor: number): Tensor {
    return new WASMTensor(this.wasmTensor.power_scalar(power, factor));
  }

  addMultiplyScalar(factor: number, add: number): Tensor {
    return new WASMTensor(this.wasmTensor.add_multiply_scalar(factor, add));
  }

  sign(): Tensor {
    return new WASMTensor(this.wasmTensor.sign());
  }

  setValues(values: Tensor, starts: number[]): Tensor {
    if (!(values instanceof WASMTensor)) {
      throw new Error('Can only set WASM values to WASM values');
    }
    return new WASMTensor(
      this.wasmTensor.set_values(values.wasmTensor, new Uint32Array(starts))
    );
  }

  add_impl(
    th: Tensor,
    tensor: Tensor,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only add WASM tensor to WASM tensor');
    }

    return new WASMTensor(
      th.wasmTensor.addition(tensor.wasmTensor, alpha, beta)
    );
  }

  subtract_impl(
    th: Tensor,
    tensor: Tensor,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only subtract WASM tensor from WASM tensor');
    }
    return new WASMTensor(
      th.wasmTensor.subtraction(tensor.wasmTensor, alpha, beta)
    );
  }

  multiply_impl(
    th: Tensor,
    tensor: Tensor,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    resultShape: readonly number[],
    alpha: number
  ): Tensor {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only multiply WASM tensor with WASM tensor');
    }
    return new WASMTensor(th.wasmTensor.multiply(tensor.wasmTensor, alpha));
  }

  divide_impl(
    th: Tensor,
    tensor: Tensor,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    resultShape: readonly number[],
    alpha: number
  ): Tensor {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only divide WASM tensor by WASM tensor');
    }
    return new WASMTensor(th.wasmTensor.divide(tensor.wasmTensor, alpha));
  }

  power_impl(
    th: Tensor,
    tensor: Tensor,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    resultShape: readonly number[]
  ): Tensor {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only take WASM tensor to power of WASM tensor');
    }
    return new WASMTensor(th.wasmTensor.power(tensor.wasmTensor));
  }

  matMul(tensor: Tensor): Tensor {
    if (!(tensor instanceof WASMTensor)) {
      throw new Error('Can only add WASM tensor to WASM tensor');
    }
    return new WASMTensor(this.wasmTensor.matmul(tensor.wasmTensor));
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
      !(b instanceof WASMTensor && (c === undefined || c instanceof WASMTensor))
    ) {
      throw new Error('Can only do gemm with CPU tensors');
    }
    if (c !== undefined) {
      return new WASMTensor(
        this.wasmTensor.gemm_with_c(
          b.wasmTensor,
          aTranspose,
          bTranspose,
          alpha,
          (c as WASMTensor).wasmTensor,
          beta
        )
      );
    } else {
      return new WASMTensor(
        this.wasmTensor.gemm(b.wasmTensor, aTranspose, bTranspose, alpha)
      );
    }
  }

  sum_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(this.wasmTensor.sum(new Uint32Array(axes), keepDims));
  }

  sumSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(
      this.wasmTensor.sum_square(new Uint32Array(axes), keepDims)
    );
  }

  product_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(
      this.wasmTensor.product(new Uint32Array(axes), keepDims)
    );
  }

  max_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(this.wasmTensor.max(new Uint32Array(axes), keepDims));
  }

  min_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(this.wasmTensor.min(new Uint32Array(axes), keepDims));
  }

  reduceMean_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(
      this.wasmTensor.reduce_mean(new Uint32Array(axes), keepDims)
    );
  }

  reduceMeanSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(
      this.wasmTensor.reduce_mean_square(new Uint32Array(axes), keepDims)
    );
  }

  protected reduceLogSum_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(
      this.wasmTensor.reduce_log_sum(new Uint32Array(axes), keepDims)
    );
  }

  protected reduceLogSumExp_impl(axes: number[], keepDims: boolean): Tensor {
    return new WASMTensor(
      this.wasmTensor.reduce_log_sum_exp(new Uint32Array(axes), keepDims)
    );
  }

  getActivationFlag(activation: Activation) {
    if (activation === 'id') {
      return 0;
    } else if (activation === 'relu') {
      return 1;
    } else {
      return 2;
    }
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
      !(kernel instanceof WASMTensor) ||
      (bias !== undefined && !(bias instanceof WASMTensor))
    ) {
      throw new Error(
        'Can only do convolution of WASM tensor with WASM tensor'
      );
    }

    const activationFlag = this.getActivationFlag(activation);

    if (bias !== undefined) {
      return new WASMTensor(
        this.wasmTensor.conv_with_bias(
          kernel.wasmTensor,
          (bias as WASMTensor).wasmTensor,
          new Uint32Array(dilations),
          group,
          new Uint32Array(pads),
          new Uint32Array(strides),
          activationFlag
        )
      );
    } else {
      return new WASMTensor(
        this.wasmTensor.conv(
          kernel.wasmTensor,
          new Uint32Array(dilations),
          group,
          new Uint32Array(pads),
          new Uint32Array(strides),
          activationFlag
        )
      );
    }
  }

  protected convTranspose_impl(
    kernel: Tensor,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[]
  ): Tensor {
    if (!(kernel instanceof WASMTensor)) {
      throw new Error(
        'Can only do transpose convolution of WASM tensor with WASM tensor'
      );
    }

    return new WASMTensor(
      this.wasmTensor.conv_transpose(
        kernel.wasmTensor,
        new Uint32Array(dilations),
        group,
        new Uint32Array(pads),
        new Uint32Array(strides)
      )
    );
  }

  averagePool_impl(
    kernelShape: number[],
    pads: number[],
    strides: number[],
    includePad: boolean
  ): Tensor {
    return new WASMTensor(
      this.wasmTensor.average_pool(
        new Uint32Array(kernelShape),
        new Uint32Array(pads),
        new Uint32Array(strides),
        includePad
      )
    );
  }

  reshape_impl(shape: number[]): Tensor {
    const sh = new Uint32Array(shape);
    return new WASMTensor(this.wasmTensor.reshape(sh), sh);
  }

  concat(tensor: Tensor, axis: number): Tensor {
    if (!(tensor instanceof WASMTensor)) {
      throw new Error('Can only concat WASM tensor to WASM tensor');
    }
    if (axis < 0) {
      axis += this.getShape().length;
    }
    return new WASMTensor(this.wasmTensor.concat(tensor.wasmTensor, axis));
  }

  transpose_impl(permutation: number[]): Tensor {
    return new WASMTensor(
      this.wasmTensor.transpose(new Uint32Array(permutation))
    );
  }

  clip(min?: number, max?: number): Tensor {
    if (min !== undefined && max !== undefined) {
      return new WASMTensor(this.wasmTensor.clip(min, max));
    } else if (min !== undefined) {
      return new WASMTensor(this.wasmTensor.clip_min(min));
    } else if (max !== undefined) {
      return new WASMTensor(this.wasmTensor.clip_max(max));
    }
    return this.copy();
  }

  clipBackward(grad: Tensor, min?: number, max?: number): Tensor {
    if (!(grad instanceof WASMTensor)) {
      throw new Error('Can only do grad backward with Wasm tensor');
    }
    if (min !== undefined && max !== undefined) {
      return new WASMTensor(
        this.wasmTensor.clip_backward(min, max, grad.wasmTensor)
      );
    } else if (min !== undefined) {
      return new WASMTensor(
        this.wasmTensor.clip_min_backward(min, grad.wasmTensor)
      );
    } else if (max !== undefined) {
      return new WASMTensor(
        this.wasmTensor.clip_max_backward(max, grad.wasmTensor)
      );
    }
    return this.copy();
  }

  repeat(repeats: number[]): Tensor {
    return new WASMTensor(this.wasmTensor.repeat(new Uint32Array(repeats)));
  }

  expand(shape: readonly number[]): Tensor {
    const thisShape = this.getShape();

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [_shape, goal, resultShape] = this.alignShapes(thisShape, shape);
    if (compareShapes(thisShape, resultShape)) {
      return this.copy();
    }

    const reshaped = this.reshape(_shape, false) as WASMTensor;

    return new WASMTensor(
      reshaped.wasmTensor.expand(new Uint32Array(resultShape))
    );
  }

  static padModeToInt = {
    constant: 0,
    reflect: 1,
    edge: 2,
  };

  pad_impl(pads: number[], mode: PadMode, value: number): Tensor {
    return new WASMTensor(
      this.wasmTensor.pad(
        new Uint32Array(pads),
        WASMTensor.padModeToInt[mode],
        value
      )
    );
  }

  gather(axis: number, indices: CPUTensor): Tensor {
    return new WASMTensor(
      this.wasmTensor.gather(
        axis,
        indices.values as Int32Array,
        new Uint32Array(indices.shape)
      )
    );
  }

  floor(): Tensor {
    return new WASMTensor(this.wasmTensor.floor());
  }

  ceil(): Tensor {
    return new WASMTensor(this.wasmTensor.ceil());
  }

  round(): Tensor {
    return new WASMTensor(this.wasmTensor.round());
  }

  slice_impl(
    starts: number[],
    ends: number[],
    axes: number[],
    steps: number[]
  ): Tensor {
    return new WASMTensor(
      this.wasmTensor.slice(
        new Uint32Array(starts),
        new Uint32Array(ends),
        new Uint32Array(axes),
        new Int32Array(steps)
      )
    );
  }

  upsample(scales: number[]): Tensor {
    return new WASMTensor(this.wasmTensor.upsample(new Float32Array(scales)));
  }

  normalize(
    mean: Tensor,
    variance: Tensor,
    epsilon: number,
    scale: Tensor,
    bias: Tensor
  ): Tensor {
    if (
      !(mean instanceof WASMTensor) ||
      !(variance instanceof WASMTensor) ||
      !(scale instanceof WASMTensor) ||
      !(bias instanceof WASMTensor)
    ) {
      throw new Error('Can only normalize with CPU tensors');
    }
    return new WASMTensor(
      this.wasmTensor.normalize(
        mean.wasmTensor,
        variance.wasmTensor,
        epsilon,
        scale.wasmTensor,
        bias.wasmTensor
      )
    );
  }
}
