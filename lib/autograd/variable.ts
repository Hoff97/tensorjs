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
import {ExpandBack} from './ops/util/expandBack';
import {ReshapeBack} from './ops/util/reshapeBack';
import {AddBack} from './ops/binary/addBack';
import {SubtractBack} from './ops/binary/subtractBack';
import {MultiplyBack} from './ops/binary/multiplyBack';
import {ConvBack} from './ops/conv/convBack';

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
      this.grad !== undefined ? this.grad.reshape(shape, copy) : undefined,
      new ReshapeBack(this)
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
    return new Variable(
      this.value.expand(shape),
      undefined,
      new ExpandBack(this, shape)
    );
  }

  copy(): Tensor {
    return new Variable(
      this.value.copy(),
      this.grad !== undefined ? this.grad.copy() : undefined
    );
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
    return new Variable(
      th.value.add_impl(th.value, tensor.value, resultShape),
      undefined,
      new AddBack(th, tensor, resultShape)
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
    return new Variable(
      th.value.subtract_impl(th.value, tensor.value, resultShape),
      undefined,
      new SubtractBack(th, tensor, resultShape)
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
    return new Variable(
      th.value.multiply_impl(th.value, tensor.value, resultShape),
      undefined,
      new MultiplyBack(th, tensor, resultShape)
    );
  }

  divide_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[]
  ): Tensor {
    throw new Error('Method not implemented.');
  }

  power_impl(
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

    return new Variable(
      this.value.conv(
        kernel.value,
        bias !== undefined ? bias.value : undefined,
        dilations,
        group,
        pads,
        strides
      ),
      undefined,
      new ConvBack(this, kernel, strides, pads, dilations, group, bias)
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
