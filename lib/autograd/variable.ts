/* eslint-disable @typescript-eslint/no-unused-vars */
import {Tensor} from '../library';
import {CPUTensor} from '../tensor/cpu/tensor';
import {TensorValues, Activation, PadMode, DType} from '../types';
import {AbsBack} from './ops/unary/absBack';
import {ExpBack} from './ops/unary/expBack';
import {LogBack} from './ops/unary/logBack';
import {MatMulBack} from './ops/matMul/matMulBack';
import {NegateBack} from './ops/unary/negateBack';
import {SqrtBack} from './ops/unary/sqrtBack';
import {BackwardOp, VariableI} from './types';
import {ConcatBack} from './ops/util/concatBack';
import {ClipBack} from './ops/unary/clipBack';
import {RepeatBack} from './ops/util/repeatBack';
import {ExpandBack} from './ops/util/expandBack';
import {ReshapeBack} from './ops/util/reshapeBack';
import {AddBack} from './ops/binary/addBack';
import {SubtractBack} from './ops/binary/subtractBack';
import {MultiplyBack} from './ops/binary/multiplyBack';
import {ConvBack} from './ops/conv/convBack';
import {DivideBack} from './ops/binary/divideBack';
import {PowerBack} from './ops/binary/powerBack';
import {GemmBack} from './ops/matMul/gemmBack';
import {TransposeBack} from './ops/util/transposeBack';
import {SumBack} from './ops/reduce/sumBack';
import {SumSquareBack} from './ops/reduce/sumSquareBack';
import {AddMultiplyScalarBack} from './ops/unary/addMultiplyScalarBack';
import {MeanBack} from './ops/reduce/meanBack';
import {MeanSquareBack} from './ops/reduce/meanSquareBack';
import {SliceBack} from './ops/util/sliceBack';
import {AveragePoolBack} from './ops/conv/averagePoolBack';
import {PadBack} from './ops/conv/padBack';
import {ProductBack} from './ops/reduce/productBack';
import {SigmoidBack} from './ops/unary/sigmoidBack';
import {Backend} from '../util/convert';
import {WASMTensor} from '../tensor/wasm/tensor';
import {GPUTensor} from '../tensor/gpu/tensor';
import REGL from 'regl';
import {ASinBack, ASinHBack, SinBack, SinHBack} from './ops/unary/sinBack';
import {ACosBack, ACosHBack, CosBack, CosHBack} from './ops/unary/cosBack';
import {ATanBack, ATanHBack, TanBack, TanHBack} from './ops/unary/tanBack';
import {LogSumBack} from './ops/reduce/logSumBack';
import {LogSumExpBack} from './ops/reduce/logSumExpBack';
import {PowerScalarBack} from './ops/unary/powerScalarBack';

export interface VariableOptions<DTpe extends DType> {
  /**
   * The gradient can optionally be specified
   */
  grad?: Tensor<DTpe>;

  /**
   * Backward edge of this variable
   *
   * You most likely do not want to use this
   */
  backEdge?: BackwardOp<DTpe>;

  /**
   * When set to true, gradients will not be tracked for this
   * variable. Useful for data that is passed into a model.
   */
  noGrad?: boolean;
}

/**
 * Tensor that also has a gradient associated to it
 * When noGrad is false, a dynamic computation graph on
 * this variable will be build.
 *
 * Once backward on a scalar variable (eg. a variable with shape [1])
 * is called, the gradients for all variables will be computed
 */
export class Variable<DTpe extends DType = 'float32'>
  extends Tensor<DTpe>
  implements VariableI<DTpe> {
  public grad?: Tensor<DTpe>;

  public backEdge?: BackwardOp<DTpe>;

  public noGrad: boolean;

  /**
   * Creates a variable whose value is the specified value
   */
  constructor(public value: Tensor<DTpe>, options?: VariableOptions<DTpe>) {
    super(value.dtype);

    if (options === undefined) {
      options = {};
    }

    this.grad = options.grad;

    if (options.backEdge !== undefined) {
      this.backEdge = options.backEdge;
    }

    this.noGrad = options.noGrad || false;
  }

  static create<DTpe extends DType>(
    shape: ReadonlyArray<number>,
    values: number[],
    backend: Backend,
    options?: VariableOptions<DTpe>,
    dtype?: DTpe
  ) {
    if (dtype === undefined) {
      dtype = 'float32' as any;
    }

    let value: Tensor<DTpe>;
    if (backend === 'CPU') {
      value = new CPUTensor(shape, values);
    } else if (backend === 'WASM') {
      value = new WASMTensor(
        values,
        new Uint32Array(shape),
        dtype as any
      ) as any;
    } else {
      value = new GPUTensor(values, shape, dtype as any) as any;
    }

    return new Variable(value, options);
  }

  /**
   * Creates a GPU variable from texture data (eg. Image/Video element)
   */
  static fromData(
    data: REGL.TextureImageData,
    options?: VariableOptions<'float32'>
  ): Variable<'float32'> {
    const tensor = GPUTensor.fromData(data);

    return new Variable(tensor, options);
  }

  cast<DTpe2 extends DType>(dtype: DTpe2): Tensor<DTpe2> {
    throw new Error('Method not implemented.');
  }

  /**
   * Performs a backward pass and returns wether the grad is needed or can be deleted
   */
  backward(grad?: Tensor<DTpe>): boolean {
    if (grad === undefined) {
      const ownShape = this.value.getShape();
      if (ownShape.length === 1 && ownShape[0] === 1) {
        grad = this.value;
      } else {
        throw new Error(
          'Backward without an input gradient can only be done for tensors with shape [1]'
        );
      }
    }

    let needed = false;

    if (this.grad !== undefined) {
      const oldGrad = this.grad;
      this.grad = this.grad.add(grad);
      oldGrad.delete();
    } else {
      this.grad = grad;
      needed = true;
    }

    if (this.backEdge !== undefined) {
      this.backEdge.backward(grad);
    }
    return needed;
  }

  isLeaf() {
    return this.backEdge === undefined;
  }

  constantLike(value: number): Tensor<DTpe> {
    return new Variable(this.value.constantLike(value), {noGrad: true});
  }

  singleConstant(value: number): Tensor<DTpe> {
    return new Variable(this.value.singleConstant(value), {noGrad: true});
  }

  getValues(): Promise<TensorValues[DTpe]> {
    return this.value.getValues();
  }

  getShape(): readonly number[] {
    return this.value.getShape();
  }

  delete(): void {
    this.value.delete();
    if (this.grad !== undefined) {
      this.grad.delete();
    }
    if (this.backEdge !== undefined) {
      this.backEdge.delete();
    }
  }

  protected reshape_impl(
    shape: readonly number[],
    copy: boolean
  ): Tensor<DTpe> {
    return new Variable(this.value.reshape(shape), {
      backEdge: this.noGrad ? undefined : new ReshapeBack(this),
      noGrad: this.noGrad,
    });
  }

  exp(): Tensor<DTpe> {
    const exp = this.value.exp();
    return new Variable(exp, {
      backEdge: this.noGrad ? undefined : new ExpBack(this, exp),
      noGrad: this.noGrad,
    });
  }

  log(): Tensor<DTpe> {
    return new Variable(this.value.log(), {
      backEdge: this.noGrad ? undefined : new LogBack(this),
      noGrad: this.noGrad,
    });
  }

  sqrt(): Tensor<DTpe> {
    const sqrt = this.value.sqrt();
    return new Variable(sqrt, {
      backEdge: this.noGrad ? undefined : new SqrtBack(this, sqrt),
      noGrad: this.noGrad,
    });
  }

  abs(): Tensor<DTpe> {
    return new Variable(this.value.abs(), {
      backEdge: this.noGrad ? undefined : new AbsBack(this),
      noGrad: this.noGrad,
    });
  }

  sin(): Tensor<DTpe> {
    return new Variable(this.value.sin(), {
      backEdge: this.noGrad ? undefined : new SinBack(this),
      noGrad: this.noGrad,
    });
  }

  cos(): Tensor<DTpe> {
    return new Variable(this.value.cos(), {
      backEdge: this.noGrad ? undefined : new CosBack(this),
      noGrad: this.noGrad,
    });
  }

  tan(): Tensor<DTpe> {
    return new Variable(this.value.tan(), {
      backEdge: this.noGrad ? undefined : new TanBack(this),
      noGrad: this.noGrad,
    });
  }

  asin(): Tensor<DTpe> {
    return new Variable(this.value.asin(), {
      backEdge: this.noGrad ? undefined : new ASinBack(this),
      noGrad: this.noGrad,
    });
  }

  acos(): Tensor<DTpe> {
    return new Variable(this.value.acos(), {
      backEdge: this.noGrad ? undefined : new ACosBack(this),
      noGrad: this.noGrad,
    });
  }

  atan(): Tensor<DTpe> {
    return new Variable(this.value.atan(), {
      backEdge: this.noGrad ? undefined : new ATanBack(this),
      noGrad: this.noGrad,
    });
  }

  sinh(): Tensor<DTpe> {
    return new Variable(this.value.sinh(), {
      backEdge: this.noGrad ? undefined : new SinHBack(this),
      noGrad: this.noGrad,
    });
  }

  cosh(): Tensor<DTpe> {
    return new Variable(this.value.cosh(), {
      backEdge: this.noGrad ? undefined : new CosHBack(this),
      noGrad: this.noGrad,
    });
  }

  tanh(): Tensor<DTpe> {
    const tanh = this.value.tanh();
    return new Variable(tanh, {
      backEdge: this.noGrad ? undefined : new TanHBack(this, tanh),
      noGrad: this.noGrad,
    });
  }

  asinh(): Tensor<DTpe> {
    return new Variable(this.value.asinh(), {
      backEdge: this.noGrad ? undefined : new ASinHBack(this),
      noGrad: this.noGrad,
    });
  }

  acosh(): Tensor<DTpe> {
    return new Variable(this.value.acosh(), {
      backEdge: this.noGrad ? undefined : new ACosHBack(this),
      noGrad: this.noGrad,
    });
  }

  atanh(): Tensor<DTpe> {
    return new Variable(this.value.atanh(), {
      backEdge: this.noGrad ? undefined : new ATanHBack(this),
      noGrad: this.noGrad,
    });
  }

  sigmoid(): Tensor<DTpe> {
    const sigmoid = this.value.sigmoid();
    return new Variable(sigmoid, {
      backEdge: this.noGrad ? undefined : new SigmoidBack(this, sigmoid),
      noGrad: this.noGrad,
    });
  }

  hardSigmoid(alpha: number, beta: number): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  sign(): Tensor<DTpe> {
    // No back edge since the gradient will be zero anyway
    return new Variable(this.value.sign());
  }

  negate(): Tensor<DTpe> {
    return new Variable(this.value.negate(), {
      backEdge: this.noGrad ? undefined : new NegateBack(this),
      noGrad: this.noGrad,
    });
  }

  addMultiplyScalar(factor: number, add: number): Tensor<DTpe> {
    return new Variable(this.value.addMultiplyScalar(factor, add), {
      backEdge: this.noGrad
        ? undefined
        : new AddMultiplyScalarBack(this, factor),
      noGrad: this.noGrad,
    });
  }

  powerScalar(power: number, factor: number): Tensor<DTpe> {
    return new Variable(this.value.powerScalar(power, factor), {
      backEdge: this.noGrad
        ? undefined
        : new PowerScalarBack(this, power, factor),
      noGrad: this.noGrad,
    });
  }

  setValues(values: Tensor<DTpe>, starts: number[]): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  matMul(tensor: Tensor<DTpe>): Tensor<DTpe> {
    if (!(tensor instanceof Variable)) {
      throw new Error('MatMul can only be done with another variable');
    }

    const noGrad = this.noGrad && tensor.noGrad;

    return new Variable(this.value.matMul(tensor.value), {
      backEdge: noGrad ? undefined : new MatMulBack(this, tensor),
      noGrad,
    });
  }

  concat(tensor: Tensor<DTpe>, axis: number): Tensor<DTpe> {
    if (!(tensor instanceof Variable)) {
      throw new Error('Concat can only be done with another variable');
    }

    const noGrad = this.noGrad && tensor.noGrad;

    return new Variable(this.value.concat(tensor.value, axis), {
      backEdge: noGrad ? undefined : new ConcatBack(this, tensor, axis),
      noGrad,
    });
  }

  clip(min?: number, max?: number): Tensor<DTpe> {
    return new Variable(this.value.clip(min, max), {
      backEdge: this.noGrad ? undefined : new ClipBack(this, min, max),
      noGrad: this.noGrad,
    });
  }

  clipBackward(grad: Tensor<DTpe>, min?: number, max?: number): Tensor<DTpe> {
    throw new Error('Clip backward not implemented for Variable');
  }

  repeat(repeats: number[]): Tensor<DTpe> {
    return new Variable(this.value.repeat(repeats), {
      backEdge: this.noGrad ? undefined : new RepeatBack(this, repeats),
      noGrad: this.noGrad,
    });
  }

  expand(shape: readonly number[]): Tensor<DTpe> {
    return new Variable(this.value.expand(shape), {
      backEdge: this.noGrad ? undefined : new ExpandBack(this, shape),
      noGrad: this.noGrad,
    });
  }

  copy(): Tensor<DTpe> {
    return new Variable(this.value.copy(), {
      grad: this.grad !== undefined ? this.grad.copy() : undefined,
    });
  }

  gather(axis: number, indices: CPUTensor<'uint32'>): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  floor(): Tensor<DTpe> {
    return new Variable(this.value.floor());
  }

  ceil(): Tensor<DTpe> {
    return new Variable(this.value.ceil());
  }

  round(): Tensor<DTpe> {
    return new Variable(this.value.round());
  }

  upsample(scales: number[]): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  normalize(
    mean: Tensor<DTpe>,
    variance: Tensor<DTpe>,
    epsilon: number,
    scale: Tensor<DTpe>,
    bias: Tensor<DTpe>
  ): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  add_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof Variable) || !(th instanceof Variable)) {
      throw new Error('Can only add Variable tensor to Variable tensor');
    }

    const noGrad = th.noGrad && tensor.noGrad;

    return new Variable(
      th.value.add_impl(th.value, tensor.value, resultShape, alpha, beta),
      {
        backEdge: noGrad
          ? undefined
          : new AddBack(th, tensor, resultShape, alpha, beta),
        noGrad,
      }
    );
  }

  subtract_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof Variable) || !(th instanceof Variable)) {
      throw new Error('Can only add Variable tensor to Variable tensor');
    }

    const noGrad = th.noGrad && tensor.noGrad;

    return new Variable(
      th.value.subtract_impl(th.value, tensor.value, resultShape, alpha, beta),
      {
        backEdge: noGrad
          ? undefined
          : new SubtractBack(th, tensor, resultShape, alpha, beta),
        noGrad,
      }
    );
  }

  multiply_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof Variable) || !(th instanceof Variable)) {
      throw new Error('Can only add Variable tensor to Variable tensor');
    }

    const noGrad = th.noGrad && tensor.noGrad;

    return new Variable(
      th.value.multiply_impl(th.value, tensor.value, resultShape, alpha),
      {
        backEdge: noGrad
          ? undefined
          : new MultiplyBack(th, tensor, resultShape, alpha),
        noGrad,
      }
    );
  }

  divide_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number
  ): Tensor<DTpe> {
    if (!(tensor instanceof Variable) || !(th instanceof Variable)) {
      throw new Error('Can only divide Variable tensor by Variable tensor');
    }

    const divResult = th.value.divide_impl(
      th.value,
      tensor.value,
      resultShape,
      alpha
    );
    const noGrad = th.noGrad && tensor.noGrad;

    return new Variable(divResult, {
      backEdge: noGrad
        ? undefined
        : new DivideBack(th, tensor, divResult, resultShape, alpha),
      noGrad,
    });
  }

  power_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[]
  ): Tensor<DTpe> {
    if (!(tensor instanceof Variable) || !(th instanceof Variable)) {
      throw new Error(
        'Can only take Variable tensor to power of Variable tensor'
      );
    }

    const powerResult = th.value.power_impl(
      th.value,
      tensor.value,
      resultShape
    );

    const noGrad = th.noGrad && tensor.noGrad;

    return new Variable(powerResult, {
      backEdge: noGrad
        ? undefined
        : new PowerBack(th, tensor, powerResult, resultShape),
      noGrad,
    });
  }

  gemm_impl(
    b: Tensor<DTpe>,
    aTranspose: boolean,
    bTranspose: boolean,
    alpha: number,
    beta: number,
    C?: Tensor<DTpe>
  ): Tensor<DTpe> {
    if (
      !(b instanceof Variable) ||
      (C !== undefined && !(C instanceof Variable))
    ) {
      throw new Error('Can only do gemm with variable tensors');
    }

    const noGrad =
      this.noGrad && b.noGrad && (C !== undefined ? C.noGrad : true);

    return new Variable(
      this.value.gemm_impl(
        b.value,
        aTranspose,
        bTranspose,
        alpha,
        beta,
        C !== undefined ? C.value : undefined
      ),
      {
        backEdge: noGrad
          ? undefined
          : new GemmBack(this, b, aTranspose, bTranspose, alpha, beta, C),
        noGrad,
      }
    );
  }

  protected sum_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return new Variable(this.value.sum(axes, keepDims), {
      backEdge: this.noGrad ? undefined : new SumBack(this, axes, keepDims),
      noGrad: this.noGrad,
    });
  }

  protected sumSquare_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return new Variable(this.value.sumSquare(axes, keepDims), {
      backEdge: this.noGrad
        ? undefined
        : new SumSquareBack(this, axes, keepDims),
      noGrad: this.noGrad,
    });
  }

  protected product_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    const product = this.value.product(axes, keepDims);
    return new Variable(product, {
      backEdge: this.noGrad
        ? undefined
        : new ProductBack(this, product, axes, keepDims),
      noGrad: this.noGrad,
    });
  }

  protected max_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  protected argMax_impl(axes: number[], selectLast: boolean): Tensor<'uint32'> {
    return new Variable(this.value.argMax(axes, selectLast));
  }

  protected min_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  protected reduceMean_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return new Variable(this.value.reduceMean(axes, keepDims), {
      backEdge: this.noGrad ? undefined : new MeanBack(this, axes, keepDims),
      noGrad: this.noGrad,
    });
  }

  protected reduceMeanSquare_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor<DTpe> {
    return new Variable(this.value.reduceMeanSquare(axes, keepDims), {
      backEdge: this.noGrad
        ? undefined
        : new MeanSquareBack(this, axes, keepDims),
      noGrad: this.noGrad,
    });
  }

  protected reduceLogSum_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    return new Variable(this.value.reduceLogSum(axes, keepDims), {
      backEdge: this.noGrad ? undefined : new LogSumBack(this, axes, keepDims),
      noGrad: this.noGrad,
    });
  }

  protected reduceLogSumExp_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor<DTpe> {
    return new Variable(this.value.reduceLogSumExp(axes, keepDims), {
      backEdge: this.noGrad
        ? undefined
        : new LogSumExpBack(this, axes, keepDims),
      noGrad: this.noGrad,
    });
  }

  protected conv_impl(
    kernel: Tensor<DTpe>,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[],
    activation: Activation,
    bias?: Tensor<DTpe>
  ): Tensor<DTpe> {
    if (
      !(kernel instanceof Variable) ||
      (bias !== undefined && !(bias instanceof Variable))
    ) {
      throw new Error(
        'Can only do convolution with variable as kernel and bias'
      );
    }

    if (activation !== 'id') {
      throw new Error('Activation has to be ID for convolution with variables');
    }

    const noGrad =
      this.noGrad && kernel.noGrad && (bias !== undefined ? bias.noGrad : true);

    return new Variable(
      this.value.conv(
        kernel.value,
        bias !== undefined ? bias.value : undefined,
        dilations,
        group,
        pads,
        strides
      ),
      {
        backEdge: noGrad
          ? undefined
          : new ConvBack(this, kernel, strides, pads, dilations, group, bias),
        noGrad,
      }
    );
  }

  protected convTranspose_impl(
    kernel: Tensor<DTpe>,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[]
  ): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  protected pad_impl(
    pads: number[],
    mode: PadMode,
    value: number
  ): Tensor<DTpe> {
    return new Variable(this.value.pad(pads, mode, value), {
      backEdge: this.noGrad ? undefined : new PadBack(this, pads, mode, value),
      noGrad: this.noGrad,
    });
  }

  protected averagePool_impl(
    kernelShape: number[],
    pads: number[],
    strides: number[],
    includePad: boolean
  ): Tensor<DTpe> {
    return new Variable(
      this.value.averagePool(kernelShape, pads, strides, includePad),
      {
        backEdge: this.noGrad
          ? undefined
          : new AveragePoolBack(this, kernelShape, pads, strides, includePad),
        noGrad: this.noGrad,
      }
    );
  }

  protected transpose_impl(permutation: number[]): Tensor<DTpe> {
    return new Variable(this.value.transpose(permutation), {
      backEdge: this.noGrad ? undefined : new TransposeBack(this, permutation),
      noGrad: this.noGrad,
    });
  }

  protected slice_impl(
    starts: number[],
    ends: number[],
    axes: number[],
    steps: number[]
  ): Tensor<DTpe> {
    return new Variable(this.value.slice(starts, ends, axes, steps), {
      backEdge: this.noGrad
        ? undefined
        : new SliceBack(this, starts, ends, axes, steps),
      noGrad: this.noGrad,
    });
  }
}
