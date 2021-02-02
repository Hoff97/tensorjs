/* eslint-disable @typescript-eslint/no-unused-vars */
import {Tensor} from '../library';
import {CPUTensor} from '../tensor/cpu/tensor';
import {TensorValues, Activation, PadMode} from '../types';
import {AbsBack} from './ops/unary/absBack';
import {ExpBack} from './ops/unary/expBack';
import {LogBack} from './ops/unary/logBack';
import {MatMulBack} from './ops/matMulBack';
import {NegateBack} from './ops/unary/negateBack';
import {SqrtBack} from './ops/unary/sqrtBack';
import {BackwardOp, VariableI} from './types';
import {ConcatBack} from './ops/util/concatBack';
import {ClipBack} from './ops/unary/clipBack';
import {RepeatBack} from './ops/util/repeatBack';

export class Variable extends Tensor implements VariableI {
  public grad?: Tensor;

  public backEdge?: BackwardOp;

  constructor(public value: Tensor, grad?: Tensor, backEdge?: BackwardOp) {
    super();
    this.grad = grad;

    if (backEdge !== undefined) {
      this.backEdge = backEdge;
    }
  }

  backward(grad: Tensor) {
    if (this.grad !== undefined) {
      const oldGrad = this.grad;
      this.grad = this.grad.add(grad);
      oldGrad.delete();
    } else {
      this.grad = grad;
    }

    if (this.backEdge !== undefined) {
      this.backEdge.backward(this.grad);
    }
  }

  constantLike(value: number): Tensor {
    return new Variable(this.value.constantLike(value));
  }

  singleConstant(value: number): Tensor {
    return new Variable(this.value.singleConstant(value));
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
  }

  protected reshape_impl(shape: readonly number[], copy: boolean): Tensor {
    return new Variable(
      this.value.reshape(shape, copy),
      this.grad !== undefined ? this.grad.reshape(shape, copy) : undefined
    );
  }

  exp(): Tensor {
    const exp = this.value.exp();
    return new Variable(exp, undefined, new ExpBack(this, exp));
  }

  log(): Tensor {
    return new Variable(this.value.log(), undefined, new LogBack(this));
  }

  sqrt(): Tensor {
    const sqrt = this.value.sqrt();
    return new Variable(sqrt, undefined, new SqrtBack(this, sqrt));
  }

  abs(): Tensor {
    return new Variable(this.value.abs(), undefined, new AbsBack(this));
  }

  sign(): Tensor {
    // No back edge since the gradient will be zero anyway
    return new Variable(this.value.sqrt(), undefined);
  }

  negate(): Tensor {
    return new Variable(this.value.negate(), undefined, new NegateBack(this));
  }

  matMul(tensor: Tensor): Tensor {
    if (!(tensor instanceof Variable)) {
      throw new Error('MatMul can only be done with another variable');
    }

    return new Variable(
      this.value.matMul(tensor.value),
      undefined,
      new MatMulBack(this, tensor)
    );
  }

  concat(tensor: Tensor, axis: number): Tensor {
    if (!(tensor instanceof Variable)) {
      throw new Error('Concat can only be done with another variable');
    }
    return new Variable(
      this.value.concat(tensor.value, axis),
      undefined,
      new ConcatBack(this, tensor, axis)
    );
  }

  clip(min?: number, max?: number): Tensor {
    return new Variable(
      this.value.clip(min, max),
      undefined,
      new ClipBack(this, min, max)
    );
  }

  clipBackward(grad: Tensor, min?: number, max?: number): Tensor {
    throw new Error('Clip backward not implemented for Variable');
  }

  repeat(repeats: number[]): Tensor {
    return new Variable(
      this.value.repeat(repeats),
      undefined,
      new RepeatBack(this, repeats)
    );
  }

  expand(shape: number[]): Tensor {
    throw new Error('Method not implemented.');
  }

  copy(): Tensor {
    throw new Error('Method not implemented.');
  }
  gather(axis: number, indices: CPUTensor): Tensor {
    throw new Error('Method not implemented.');
  }
  floor(): Tensor {
    throw new Error('Method not implemented.');
  }
  ceil(): Tensor {
    throw new Error('Method not implemented.');
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
  protected add_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[]
  ): Tensor {
    throw new Error('Method not implemented.');
  }
  protected subtract_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[]
  ): Tensor {
    throw new Error('Method not implemented.');
  }
  protected multiply_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[]
  ): Tensor {
    throw new Error('Method not implemented.');
  }
  protected divide_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[]
  ): Tensor {
    throw new Error('Method not implemented.');
  }
  protected power_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[]
  ): Tensor {
    throw new Error('Method not implemented.');
  }
  protected gemm_impl(
    b: Tensor,
    aTranspose: boolean,
    bTranspose: boolean,
    alpha: number,
    beta: number,
    C?: Tensor
  ): Tensor {
    throw new Error('Method not implemented.');
  }
  protected sum_impl(axes: number[], keepDims: boolean): Tensor {
    throw new Error('Method not implemented.');
  }
  protected sumSquare_impl(axes: number[], keepDims: boolean): Tensor {
    throw new Error('Method not implemented.');
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
    throw new Error('Method not implemented.');
  }
  protected slice_impl(
    starts: number[],
    ends: number[],
    axes: number[]
  ): Tensor {
    throw new Error('Method not implemented.');
  }
}
