import {Activation} from '../../library';
import Tensor, {
  DType,
  PadMode,
  TensorValues,
  tensorValuesConstructor,
} from '../../types';
import {compareShapes} from '../../util/shape';

import {
  TensorF32 as WTF32,
  TensorF64 as WTF64,
  TensorI32 as WTI32,
  TensorI16 as WTI16,
  TensorI8 as WTI8,
  TensorU32 as WTU32,
  TensorU16 as WTU16,
  TensorU8 as WTU8,
} from '../../wasm/rust_wasm_tensor';
import {CPUTensor} from '../cpu/tensor';

let WASMTF32: typeof WTF32;
let WASMTF64: typeof WTF64;
let WASMTI32: typeof WTI32;
let WASMTI16: typeof WTI16;
let WASMTI8: typeof WTI8;
let WASMTU32: typeof WTU32;
let WASMTU16: typeof WTU16;
let WASMTU8: typeof WTU8;

export let tensorConstructor: {[name: string]: any};

export const wasmLoaded: Promise<void> = new Promise(resolve => {
  import('../../wasm/rust_wasm_tensor').then(x => {
    WASMTF32 = x.TensorF32;
    WASMTF64 = x.TensorF64;
    WASMTI32 = x.TensorI32;
    WASMTI16 = x.TensorI16;
    WASMTI8 = x.TensorI8;
    WASMTU32 = x.TensorU32;
    WASMTU16 = x.TensorU16;
    WASMTU8 = x.TensorU8;

    tensorConstructor = {
      float64: WASMTF32,
      float32: WASMTF64,
      int32: WASMTI32,
      int16: WASMTI16,
      int8: WASMTI8,
      uint32: WASMTU32,
      uint16: WASMTU16,
      uint8: WASMTU8,
    };

    resolve();
  });
});

export type WT = {
  float64: WTF64;
  float32: WTF32;
  int32: WTI32;
  int16: WTI16;
  int8: WTI8;
  uint32: WTU32;
  uint16: WTU16;
  uint8: WTU8;
};

export type DTypeWasm =
  | 'float64'
  | 'float32'
  | 'int32'
  | 'int16'
  | 'int8'
  | 'uint32'
  | 'uint16'
  | 'uint8';

export class WASMTensor<DTpe extends DTypeWasm> extends Tensor<DTpe> {
  static range(start: number, limit: number, delta: number) {
    const size = Math.max(Math.ceil((limit - start) / delta), 0);
    const values = new Array(size);
    for (let i = 0; i < size; i++) {
      values[i] = start + i * delta;
    }
    return new WASMTensor(values, new Uint32Array([size]));
  }

  public wasmTensor: WT[DTpe];

  constructor(values: number[] | WT[DTpe], shape?: Uint32Array, dtype?: DTpe) {
    super(dtype || ('float32' as any));

    if (values instanceof Array) {
      if (shape === undefined) {
        throw new Error(
          'Need the shape when creating a Wasm tensor from values'
        );
      }

      const array = new tensorValuesConstructor[this.dtype](
        values
      ) as TensorValues[DTpe];

      this.wasmTensor = tensorConstructor[this.dtype].create(
        shape,
        array as any
      ) as WT[DTpe];
    } else {
      this.wasmTensor = values;
    }
  }

  cast<DTpe2 extends DType>(dtype: DTpe2): Tensor<DTpe2> {
    throw new Error('Method not implemented.');
  }

  getValues() {
    // TODO: remove any
    return Promise.resolve(this.wasmTensor.get_vals()) as any;
  }

  getShape(): readonly number[] {
    return Array.from(this.wasmTensor.get_shape());
  }

  constantLike(value: number): Tensor<DTpe> {
    // TODO: Maybe more efficient in WASM?
    return new WASMTensor([value], this.wasmTensor.get_shape(), this.dtype);
  }

  singleConstant(value: number): Tensor<DTpe> {
    return new WASMTensor([value], new Uint32Array([1]), this.dtype);
  }

  delete(): void {
    if (this.wasmTensor !== undefined) {
      this.wasmTensor.free();
      //@ts-ignore
      this.wasmTensor = undefined;
    }
  }

  copy(): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.copy() as WT[DTpe],
      undefined,
      this.dtype
    );
  }

  exp(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Exp can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.exp() as WT[DTpe]);
  }

  log(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Log can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.log() as WT[DTpe]);
  }

  sqrt(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Sqrt can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.sqrt() as WT[DTpe]);
  }

  abs(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32) &&
      !(this.wasmTensor instanceof WASMTI32) &&
      !(this.wasmTensor instanceof WASMTI16) &&
      !(this.wasmTensor instanceof WASMTI8)
    ) {
      throw new Error('Abs can only be called on signed tensors');
    }
    return new WASMTensor(this.wasmTensor.abs() as WT[DTpe]);
  }

  sin(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Sin can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.sin() as WT[DTpe]);
  }

  cos(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Cos can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.cos() as WT[DTpe]);
  }

  tan(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Tan can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.tan() as WT[DTpe]);
  }

  asin(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Asin can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.asin() as WT[DTpe]);
  }

  acos(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Acos can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.acos() as WT[DTpe]);
  }

  atan(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Atan can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.atan() as WT[DTpe]);
  }

  sinh(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Sinh can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.sinh() as WT[DTpe]);
  }

  cosh(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Cosh can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.cosh() as WT[DTpe]);
  }

  tanh(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Tanh can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.tanh() as WT[DTpe]);
  }

  asinh(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Asinh can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.asinh() as WT[DTpe]);
  }

  acosh(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Acosh can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.acosh() as WT[DTpe]);
  }

  atanh(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Atanh can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.atanh() as WT[DTpe]);
  }

  sigmoid(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Sigmoid can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.sigmoid() as WT[DTpe]);
  }

  hardSigmoid(alpha: number, beta: number): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('HardSigmoid can only be called on float tensors');
    }
    return new WASMTensor(
      this.wasmTensor.hard_sigmoid(alpha, beta) as WT[DTpe]
    );
  }

  negate(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32) &&
      !(this.wasmTensor instanceof WASMTI32) &&
      !(this.wasmTensor instanceof WASMTI16) &&
      !(this.wasmTensor instanceof WASMTI8)
    ) {
      throw new Error('Negate can only be called on signed tensors');
    }
    return new WASMTensor(this.wasmTensor.negate() as WT[DTpe]);
  }

  powerScalar(power: number, factor: number): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.power_scalar(power, factor) as WT[DTpe]
    );
  }

  addMultiplyScalar(factor: number, add: number): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.add_multiply_scalar(factor, add) as WT[DTpe]
    );
  }

  sign(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32) &&
      !(this.wasmTensor instanceof WASMTI32) &&
      !(this.wasmTensor instanceof WASMTI16) &&
      !(this.wasmTensor instanceof WASMTI8)
    ) {
      throw new Error('Sign can only be called on signed tensors');
    }
    return new WASMTensor(this.wasmTensor.sign() as WT[DTpe]);
  }

  setValues(values: Tensor<DTpe>, starts: number[]): Tensor<DTpe> {
    if (!(values instanceof WASMTensor)) {
      throw new Error('Can only set WASM values to WASM values');
    }
    return new WASMTensor(
      this.wasmTensor.set_values(
        values.wasmTensor,
        new Uint32Array(starts)
      ) as WT[DTpe]
    );
  }

  add_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only add WASM tensor to WASM tensor');
    }

    return new WASMTensor(
      th.wasmTensor.addition(tensor.wasmTensor, alpha, beta)
    );
  }

  subtract_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only subtract WASM tensor from WASM tensor');
    }
    return new WASMTensor(
      th.wasmTensor.subtraction(tensor.wasmTensor, alpha, beta)
    );
  }

  multiply_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    resultShape: readonly number[],
    alpha: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only multiply WASM tensor with WASM tensor');
    }
    return new WASMTensor(th.wasmTensor.multiply(tensor.wasmTensor, alpha));
  }

  divide_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    resultShape: readonly number[],
    alpha: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only divide WASM tensor by WASM tensor');
    }
    return new WASMTensor(th.wasmTensor.divide(tensor.wasmTensor, alpha));
  }

  power_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    resultShape: readonly number[]
  ): Tensor<DTpe> {
    if (!(tensor instanceof WASMTensor) || !(th instanceof WASMTensor)) {
      throw new Error('Can only take WASM tensor to power of WASM tensor');
    }
    return new WASMTensor(th.wasmTensor.power(tensor.wasmTensor));
  }

  matMul(tensor: Tensor<DTpe>): Tensor<DTpe> {
    if (!(tensor instanceof WASMTensor)) {
      throw new Error('Can only add WASM tensor to WASM tensor');
    }
    return new WASMTensor(
      this.wasmTensor.matmul(tensor.wasmTensor) as WT[DTpe]
    );
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
          (c as WASMTensor<DTpe>).wasmTensor as any,
          beta
        ) as WT[DTpe]
      );
    } else {
      return new WASMTensor(
        this.wasmTensor.gemm(
          b.wasmTensor,
          aTranspose,
          bTranspose,
          alpha
        ) as WT[DTpe]
      );
    }
  }

  sum_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.sum(new Uint32Array(axes), keepDims) as WT[DTpe]
    );
  }

  sumSquare_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.sum_square(new Uint32Array(axes), keepDims) as WT[DTpe]
    );
  }

  product_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.product(new Uint32Array(axes), keepDims) as WT[DTpe]
    );
  }

  max_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.max(new Uint32Array(axes), keepDims) as WT[DTpe]
    );
  }

  min_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.min(new Uint32Array(axes), keepDims) as WT[DTpe]
    );
  }

  reduceMean_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.reduce_mean(new Uint32Array(axes), keepDims) as WT[DTpe]
    );
  }

  reduceMeanSquare_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.reduce_mean_square(
        new Uint32Array(axes),
        keepDims
      ) as WT[DTpe]
    );
  }

  protected reduceLogSum_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('ReduceLogSum can only be called on float tensors');
    }
    return new WASMTensor(
      this.wasmTensor.reduce_log_sum(
        new Uint32Array(axes),
        keepDims
      ) as WT[DTpe]
    );
  }

  protected reduceLogSumExp_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('ReduceLogSumExp can only be called on float tensors');
    }
    return new WASMTensor(
      this.wasmTensor.reduce_log_sum_exp(
        new Uint32Array(axes),
        keepDims
      ) as WT[DTpe]
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
    kernel: Tensor<DTpe>,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[],
    activation: Activation,
    bias?: Tensor<DTpe>
  ): Tensor<DTpe> {
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
          (bias as WASMTensor<DTpe>).wasmTensor as any,
          new Uint32Array(dilations),
          group,
          new Uint32Array(pads),
          new Uint32Array(strides),
          activationFlag
        ) as WT[DTpe]
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
        ) as WT[DTpe]
      );
    }
  }

  protected convTranspose_impl(
    kernel: Tensor<DTpe>,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[]
  ): Tensor<DTpe> {
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
      ) as WT[DTpe]
    );
  }

  averagePool_impl(
    kernelShape: number[],
    pads: number[],
    strides: number[],
    includePad: boolean
  ): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.average_pool(
        new Uint32Array(kernelShape),
        new Uint32Array(pads),
        new Uint32Array(strides),
        includePad
      ) as WT[DTpe]
    );
  }

  reshape_impl(shape: number[]): Tensor<DTpe> {
    const sh = new Uint32Array(shape);
    return new WASMTensor(this.wasmTensor.reshape(sh) as WT[DTpe], sh);
  }

  concat(tensor: Tensor<DTpe>, axis: number): Tensor<DTpe> {
    if (!(tensor instanceof WASMTensor)) {
      throw new Error('Can only concat WASM tensor to WASM tensor');
    }
    if (axis < 0) {
      axis += this.getShape().length;
    }
    return new WASMTensor(
      this.wasmTensor.concat(tensor.wasmTensor, axis) as WT[DTpe]
    );
  }

  transpose_impl(permutation: number[]): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.transpose(new Uint32Array(permutation)) as WT[DTpe]
    );
  }

  clip(min?: number, max?: number): Tensor<DTpe> {
    if (min !== undefined && max !== undefined) {
      return new WASMTensor(this.wasmTensor.clip(min, max) as WT[DTpe]);
    } else if (min !== undefined) {
      return new WASMTensor(this.wasmTensor.clip_min(min) as WT[DTpe]);
    } else if (max !== undefined) {
      return new WASMTensor(this.wasmTensor.clip_max(max) as WT[DTpe]);
    }
    return this.copy();
  }

  clipBackward(grad: Tensor<DTpe>, min?: number, max?: number): Tensor<DTpe> {
    if (!(grad instanceof WASMTensor)) {
      throw new Error('Can only do grad backward with Wasm tensor');
    }
    if (min !== undefined && max !== undefined) {
      return new WASMTensor(
        this.wasmTensor.clip_backward(min, max, grad.wasmTensor) as WT[DTpe]
      );
    } else if (min !== undefined) {
      return new WASMTensor(
        this.wasmTensor.clip_min_backward(min, grad.wasmTensor) as WT[DTpe]
      );
    } else if (max !== undefined) {
      return new WASMTensor(
        this.wasmTensor.clip_max_backward(max, grad.wasmTensor) as WT[DTpe]
      );
    }
    return this.copy();
  }

  repeat(repeats: number[]): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.repeat(new Uint32Array(repeats)) as WT[DTpe]
    );
  }

  expand(shape: readonly number[]): Tensor<DTpe> {
    const thisShape = this.getShape();

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [_shape, goal, resultShape] = this.alignShapes(thisShape, shape);
    if (compareShapes(thisShape, resultShape)) {
      return this.copy();
    }

    const reshaped = this.reshape(_shape, false) as WASMTensor<DTpe>;

    return new WASMTensor(
      reshaped.wasmTensor.expand(new Uint32Array(resultShape)) as WT[DTpe]
    );
  }

  static padModeToInt = {
    constant: 0,
    reflect: 1,
    edge: 2,
  };

  pad_impl(pads: number[], mode: PadMode, value: number): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.pad(
        new Uint32Array(pads),
        WASMTensor.padModeToInt[mode],
        value
      ) as WT[DTpe]
    );
  }

  gather(axis: number, indices: CPUTensor<'uint32'>): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.gather(
        axis,
        indices.values as Uint32Array,
        new Uint32Array(indices.shape)
      ) as WT[DTpe]
    );
  }

  floor(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Floor can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.floor() as WT[DTpe]);
  }

  ceil(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Ceil can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.ceil() as WT[DTpe]);
  }

  round(): Tensor<DTpe> {
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Round can only be called on float tensors');
    }
    return new WASMTensor(this.wasmTensor.round() as WT[DTpe]);
  }

  slice_impl(
    starts: number[],
    ends: number[],
    axes: number[],
    steps: number[]
  ): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.slice(
        new Uint32Array(starts),
        new Uint32Array(ends),
        new Uint32Array(axes),
        new Int32Array(steps)
      ) as WT[DTpe]
    );
  }

  upsample(scales: number[]): Tensor<DTpe> {
    return new WASMTensor(
      this.wasmTensor.upsample(new Float32Array(scales)) as WT[DTpe]
    );
  }

  normalize(
    mean: Tensor<DTpe>,
    variance: Tensor<DTpe>,
    epsilon: number,
    scale: Tensor<DTpe>,
    bias: Tensor<DTpe>
  ): Tensor<DTpe> {
    if (
      !(mean instanceof WASMTensor) ||
      !(variance instanceof WASMTensor) ||
      !(scale instanceof WASMTensor) ||
      !(bias instanceof WASMTensor)
    ) {
      throw new Error('Can only normalize with WASM tensors');
    }
    if (
      !(this.wasmTensor instanceof WASMTF64) &&
      !(this.wasmTensor instanceof WASMTF32)
    ) {
      throw new Error('Normalize can only be called on float tensors');
    }
    return new WASMTensor(
      (this.wasmTensor as any).normalize(
        mean.wasmTensor,
        variance.wasmTensor,
        epsilon,
        scale.wasmTensor,
        bias.wasmTensor
      )
    );
  }
}
