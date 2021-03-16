import {add} from '../../ops/sparse/binary/add/add';
import {subtract} from '../../ops/sparse/binary/subtract/subtract';
import {concat} from '../../ops/sparse/concat/concat';
import {matMul} from '../../ops/sparse/matMul/matMul';
import {repeat} from '../../ops/sparse/repeat/repeat';
import {reshape} from '../../ops/sparse/reshape/reshape';
import Tensor, {
  Activation,
  DType,
  PadMode,
  TensorValues,
  tensorValuesConstructor,
} from '../../types';
import {
  computeStrides,
  getSize,
  indexToPos,
  posToIndex,
} from '../../util/shape';
import {CPUTensor} from '../cpu/tensor';

export class SparseTensor<DTpe extends DType = 'float32'> extends Tensor<DTpe> {
  static fromDense<DTpe extends DType>(
    tensor: CPUTensor<DTpe>
  ): SparseTensor<DTpe> {
    let nnz = 0;
    const ix = [];
    const vals = [];
    for (let i = 0; i < tensor.size; i++) {
      if (tensor.get(i) !== 0) {
        nnz++;

        const index = posToIndex(i, tensor.strides);
        for (let j = 0; j < tensor.shape.length; j++) {
          ix.push(index[j]);
        }
        vals.push(tensor.get(i));
      }
    }

    const indices = new CPUTensor([nnz, tensor.shape.length], ix, 'uint32');
    const values = new CPUTensor([nnz], vals, tensor.dtype);
    return new SparseTensor(values, indices, tensor.shape);
  }

  public size: number;

  public strides: number[];

  public nnz: number;

  public sparseDims: number;

  /**
   * Creates a new sparse tensor in coordinate format. The tensor has
   * a number of sparse dimensions and optionally a number of dense
   * dimensions. The shape of a sparse tensor can thus be decomposed
   * into [...S, ...D], where S is the shape of the sparse dimensions
   * and D the shape of the dense dimensions. By default the number of
   * dense dimensions is zero
   *
   * The values tensor holds all non-zero values and has shape [NNZ, ...D]
   * where NNZ is the number of non-zero entries. The indices tensor
   * holds the location of all non-zero entries of the tensor and
   * has shape [NNZ, |S|] (where |S| is the number of sparse dimensions).
   *
   * Note that all indexes that are not specified are implicitly zero.
   * This does however **not** mean that they become non-zero on
   * certain element wise operations. Instead element wise operations
   * maintain the sparsity pattern. Otherwise, many operations would
   * create effectively dense tensors (eg. exp()), or would simply not be
   * well defined (eg. log()).
   */
  constructor(
    public values: Tensor<DTpe>,
    public indices: Tensor<'uint32'>,
    public shape: readonly number[],
    public denseDims = 0
  ) {
    super(values.dtype);

    this.size = getSize(shape);
    this.strides = computeStrides(shape);
    this.nnz = this.indices.getShape()[0];
    this.sparseDims = this.shape.length - this.denseDims;
  }

  async getValues(): Promise<TensorValues[DTpe]> {
    const vals = await this.values.getValues();
    const indices = await this.indices.getValues();

    const denseSize = getSize(this.getDenseShape(), 1);

    const sparseStrides = computeStrides(this.getSparseShape());

    const result = new tensorValuesConstructor[this.values.dtype](this.size);
    for (let i = 0; i < this.nnz; i++) {
      const sparseIx = [];
      for (let j = 0; j < this.sparseDims; j++) {
        sparseIx.push(indices[i * this.sparseDims + j]);
      }
      const sparsePos = indexToPos(sparseIx, sparseStrides);

      for (let j = 0; j < denseSize; j++) {
        result[sparsePos * denseSize + j] = vals[i * denseSize + j];
      }
    }

    return result as TensorValues[DTpe];
  }

  getSparseShape(): readonly number[] {
    return this.shape.slice(0, this.shape.length - this.denseDims);
  }

  getDenseShape(): readonly number[] {
    return this.shape.slice(this.shape.length - this.denseDims);
  }

  getShape(): readonly number[] {
    return this.shape;
  }

  /**
   * Creates a new sparse tensor with the same sparsity shape
   * and the given value everywhere.
   *
   * @param value Constant value to set at every position
   */
  constantLike(value: number): Tensor<DTpe> {
    return new SparseTensor(
      this.values.constantLike(value),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  singleConstant(value: number): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  cast<DTpe2 extends DType>(dtype: DTpe2): Tensor<DTpe2> {
    return new SparseTensor(
      this.values.cast(dtype),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  delete(): void {
    this.values.delete();
    this.indices.delete();
  }

  protected reshape_impl(
    shape: readonly number[],
    copy: boolean
  ): Tensor<DTpe> {
    return reshape(this, shape, copy);
  }

  exp(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.exp(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  log(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.log(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  sqrt(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.sqrt(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  abs(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.abs(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  sin(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.sin(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  cos(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.cos(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  tan(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.tan(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  asin(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.asin(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  acos(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.acos(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  atan(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.atan(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  sinh(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.sinh(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  cosh(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.cosh(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  tanh(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.tanh(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  asinh(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.asinh(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  acosh(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.acosh(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  atanh(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.atanh(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  negate(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.negate(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  powerScalar(power: number, factor: number): Tensor<DTpe> {
    return new SparseTensor(
      this.values.powerScalar(power, factor),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  sigmoid(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.sigmoid(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  hardSigmoid(alpha: number, beta: number): Tensor<DTpe> {
    return new SparseTensor(
      this.values.hardSigmoid(alpha, beta),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  sign(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.sign(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  addMultiplyScalar(factor: number, add: number): Tensor<DTpe> {
    return new SparseTensor(
      this.values.addMultiplyScalar(factor, add),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  /**
   * Calculates the matrix product. This tensor should have shape [M,N]
   *
   * Two cases are supported for sparse tensors:
   * - If this tensor has one sparse dimension, the resulting tensor is
   *   a sparse tensor with the same number of non-zero entries
   * - If this tensor has two sparse dimensions, the resulting tensor
   *   is dense.
   * Right now this only supports sparse-dense matrix multiplication.
   *
   * @param tensor Dense matrix to multiply with. Should have shape [N,O]
   *
   * @result Tensor with shape [M,O]
   */
  matMul(tensor: Tensor<DTpe>): Tensor<DTpe> {
    return matMul(this, tensor);
  }

  /**
   * Concatenate the two tensors along the given axis
   *
   * Note that at the moment, only concatenation along
   * sparse dimensions is supported!
   *
   */
  concat(tensor: Tensor<DTpe>, axis: number): Tensor<DTpe> {
    if (!(tensor instanceof SparseTensor)) {
      throw new Error('Can only concatenate sparse tensors!');
    }
    return concat(this, tensor, axis);
  }

  clip(min?: number, max?: number): Tensor<DTpe> {
    return new SparseTensor(
      this.values.clip(min, max),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  clipBackward(grad: Tensor<DTpe>, min?: number, max?: number): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  repeat(repeats: number[]): Tensor<DTpe> {
    return repeat(this, repeats);
  }
  expand(shape: readonly number[]): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  copy(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.copy(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  gather(axis: number, indices: CPUTensor<'uint32'>): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  setValues(values: Tensor<DTpe>, starts: number[]): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  floor(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.floor(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  ceil(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.ceil(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
  }

  round(): Tensor<DTpe> {
    return new SparseTensor(
      this.values.round(),
      this.indices.copy(),
      this.shape,
      this.denseDims
    );
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
    return add(th as SparseTensor<DTpe>, tensor, resultShape, alpha, beta);
  }

  subtract_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor<DTpe> {
    return subtract(th as SparseTensor<DTpe>, tensor, resultShape, alpha, beta);
  }

  multiply_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number
  ): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  divide_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number
  ): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  power_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[]
  ): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  gemm_impl(
    b: Tensor<DTpe>,
    aTranspose: boolean,
    bTranspose: boolean,
    alpha: number,
    beta: number,
    C?: Tensor<DTpe>
  ): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  protected sum_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  protected sumSquare_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  protected product_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  protected max_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  protected min_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  protected reduceMean_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  protected reduceMeanSquare_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  protected reduceLogSum_impl(axes: number[], keepDims: boolean): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  protected reduceLogSumExp_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor<DTpe> {
    throw new Error('Method not implemented.');
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
    throw new Error('Method not implemented.');
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
    throw new Error('Method not implemented.');
  }

  protected averagePool_impl(
    kernelShape: number[],
    pads: number[],
    strides: number[],
    includePad: boolean
  ): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  protected transpose_impl(permutation: number[]): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }

  protected slice_impl(
    starts: number[],
    ends: number[],
    axes: number[],
    steps: number[]
  ): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
}
