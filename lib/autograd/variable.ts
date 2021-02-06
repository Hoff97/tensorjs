/* eslint-disable @typescript-eslint/no-unused-vars */
import {Tensor} from '../library';
import {CPUTensor} from '../tensor/cpu/tensor';
import {TensorValues, Activation, PadMode} from '../types';
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
import {throws} from 'assert';

export interface VariableOptions {
  grad?: Tensor;
  backEdge?: BackwardOp;
  noGrad?: boolean;
}

export class Variable extends Tensor implements VariableI {
  public grad?: Tensor;

  public backEdge?: BackwardOp;

  public noGrad: boolean;

  constructor(public value: Tensor, options?: VariableOptions) {
    super();

    if (options === undefined) {
      options = {};
    }

    this.grad = options.grad;

    if (options.backEdge !== undefined) {
      this.backEdge = options.backEdge;
    }

    this.noGrad = options.noGrad || false;
  }

  backward(grad?: Tensor) {
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

    if (this.grad !== undefined) {
      const oldGrad = this.grad;
      this.grad = this.grad.add(grad);
      oldGrad.delete();
    } else {
      this.grad = grad;
    }

    if (this.backEdge !== undefined) {
      this.backEdge.backward(grad);
    }
  }

  isLeaf() {
    return this.backEdge === undefined;
  }

  constantLike(value: number): Tensor {
    return new Variable(this.value.constantLike(value), {noGrad: true});
  }

  singleConstant(value: number): Tensor {
    return new Variable(this.value.singleConstant(value), {noGrad: true});
  }

  getValues(): Promise<TensorValues> {
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

  protected reshape_impl(shape: readonly number[], copy: boolean): Tensor {
    return new Variable(this.value.reshape(shape, copy), {
      backEdge: this.noGrad ? undefined : new ReshapeBack(this),
      noGrad: this.noGrad,
    });
  }

  exp(): Tensor {
    const exp = this.value.exp();
    return new Variable(exp, {
      backEdge: this.noGrad ? undefined : new ExpBack(this, exp),
      noGrad: this.noGrad,
    });
  }

  log(): Tensor {
    return new Variable(this.value.log(), {
      backEdge: this.noGrad ? undefined : new LogBack(this),
      noGrad: this.noGrad,
    });
  }

  sqrt(): Tensor {
    const sqrt = this.value.sqrt();
    return new Variable(sqrt, {
      backEdge: this.noGrad ? undefined : new SqrtBack(this, sqrt),
      noGrad: this.noGrad,
    });
  }

  abs(): Tensor {
    return new Variable(this.value.abs(), {
      backEdge: this.noGrad ? undefined : new AbsBack(this),
      noGrad: this.noGrad,
    });
  }

  sign(): Tensor {
    // No back edge since the gradient will be zero anyway
    return new Variable(this.value.sqrt());
  }

  negate(): Tensor {
    return new Variable(this.value.negate(), {
      backEdge: this.noGrad ? undefined : new NegateBack(this),
      noGrad: this.noGrad,
    });
  }

  matMul(tensor: Tensor): Tensor {
    if (!(tensor instanceof Variable)) {
      throw new Error('MatMul can only be done with another variable');
    }

    const noGrad = this.noGrad && tensor.noGrad;

    return new Variable(this.value.matMul(tensor.value), {
      backEdge: noGrad ? undefined : new MatMulBack(this, tensor),
      noGrad,
    });
  }

  concat(tensor: Tensor, axis: number): Tensor {
    if (!(tensor instanceof Variable)) {
      throw new Error('Concat can only be done with another variable');
    }

    const noGrad = this.noGrad && tensor.noGrad;

    return new Variable(this.value.concat(tensor.value, axis), {
      backEdge: noGrad ? undefined : new ConcatBack(this, tensor, axis),
      noGrad,
    });
  }

  clip(min?: number, max?: number): Tensor {
    return new Variable(this.value.clip(min, max), {
      backEdge: this.noGrad ? undefined : new ClipBack(this, min, max),
      noGrad: this.noGrad,
    });
  }

  clipBackward(grad: Tensor, min?: number, max?: number): Tensor {
    throw new Error('Clip backward not implemented for Variable');
  }

  repeat(repeats: number[]): Tensor {
    return new Variable(this.value.repeat(repeats), {
      backEdge: this.noGrad ? undefined : new RepeatBack(this, repeats),
      noGrad: this.noGrad,
    });
  }

  expand(shape: readonly number[]): Tensor {
    return new Variable(this.value.expand(shape), {
      backEdge: this.noGrad ? undefined : new ExpandBack(this, shape),
      noGrad: this.noGrad,
    });
  }

  copy(): Tensor {
    return new Variable(this.value.copy(), {
      grad: this.grad !== undefined ? this.grad.copy() : undefined,
    });
  }

  gather(axis: number, indices: CPUTensor): Tensor {
    throw new Error('Method not implemented.');
  }

  floor(): Tensor {
    return new Variable(this.value.floor());
  }

  ceil(): Tensor {
    return new Variable(this.value.ceil());
  }

  upsample(scales: number[]): Tensor {
    throw new Error('Method not implemented.');
  }

  normalize(
    mean: Tensor,
    variance: Tensor,
    epsilon: number,
    scale: Tensor,
    bias: Tensor
  ): Tensor {
    throw new Error('Method not implemented.');
  }

  add_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor {
    if (!(tensor instanceof Variable) || !(th instanceof Variable)) {
      throw new Error('Can only add Variable tensor to Variable tensor');
    }

    const noGrad = th.noGrad && tensor.noGrad;

    return new Variable(
      th.value.add_impl(th.value, tensor.value, resultShape),
      {
        backEdge: noGrad ? undefined : new AddBack(th, tensor, resultShape),
        noGrad,
      }
    );
  }

  subtract_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[]
  ): Tensor {
    if (!(tensor instanceof Variable) || !(th instanceof Variable)) {
      throw new Error('Can only add Variable tensor to Variable tensor');
    }

    const noGrad = th.noGrad && tensor.noGrad;

    return new Variable(
      th.value.subtract_impl(th.value, tensor.value, resultShape),
      {
        backEdge: noGrad
          ? undefined
          : new SubtractBack(th, tensor, resultShape),
        noGrad,
      }
    );
  }

  multiply_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[]
  ): Tensor {
    if (!(tensor instanceof Variable) || !(th instanceof Variable)) {
      throw new Error('Can only add Variable tensor to Variable tensor');
    }

    const noGrad = th.noGrad && tensor.noGrad;

    return new Variable(
      th.value.multiply_impl(th.value, tensor.value, resultShape),
      {
        backEdge: noGrad
          ? undefined
          : new MultiplyBack(th, tensor, resultShape),
        noGrad,
      }
    );
  }

  divide_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[]
  ): Tensor {
    if (!(tensor instanceof Variable) || !(th instanceof Variable)) {
      throw new Error('Can only divide Variable tensor by Variable tensor');
    }

    const divResult = th.value.divide_impl(th.value, tensor.value, resultShape);
    const noGrad = th.noGrad && tensor.noGrad;

    return new Variable(divResult, {
      backEdge: noGrad
        ? undefined
        : new DivideBack(th, tensor, divResult, resultShape),
      noGrad,
    });
  }

  power_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[]
  ): Tensor {
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
    b: Tensor,
    aTranspose: boolean,
    bTranspose: boolean,
    alpha: number,
    beta: number,
    C?: Tensor
  ): Tensor {
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

  protected sum_impl(axes: number[], keepDims: boolean): Tensor {
    return new Variable(this.value.sum(axes, keepDims), {
      backEdge: this.noGrad ? undefined : new SumBack(this, axes, keepDims),
      noGrad: this.noGrad,
    });
  }

  protected sumSquare_impl(axes: number[], keepDims: boolean): Tensor {
    return new Variable(this.value.sumSquare(axes, keepDims), {
      backEdge: this.noGrad
        ? undefined
        : new SumSquareBack(this, axes, keepDims),
      noGrad: this.noGrad,
    });
  }

  protected product_impl(axes: number[], keepDims: boolean): Tensor {
    throw new Error('Method not implemented.');
  }
  protected max_impl(axes: number[], keepDims: boolean): Tensor {
    throw new Error('Method not implemented.');
  }
  protected min_impl(axes: number[], keepDims: boolean): Tensor {
    throw new Error('Method not implemented.');
  }
  protected reduceMean_impl(axes: number[], keepDims: boolean): Tensor {
    throw new Error('Method not implemented.');
  }
  protected reduceMeanSquare_impl(axes: number[], keepDims: boolean): Tensor {
    throw new Error('Method not implemented.');
  }

  protected conv_impl(
    kernel: Tensor,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[],
    activation: Activation,
    bias?: Tensor
  ): Tensor {
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
    kernel: Tensor,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[]
  ): Tensor {
    throw new Error('Method not implemented.');
  }

  protected pad_impl(pads: number[], mode: PadMode, value: number): Tensor {
    throw new Error('Method not implemented.');
  }
  protected averagePool_impl(
    kernelShape: number[],
    pads: number[],
    strides: number[],
    includePad: boolean
  ): Tensor {
    throw new Error('Method not implemented.');
  }

  protected transpose_impl(permutation: number[]): Tensor {
    return new Variable(this.value.transpose(permutation), {
      backEdge: this.noGrad ? undefined : new TransposeBack(this, permutation),
      noGrad: this.noGrad,
    });
  }

  protected slice_impl(
    starts: number[],
    ends: number[],
    axes: number[]
  ): Tensor {
    throw new Error('Method not implemented.');
  }
}
