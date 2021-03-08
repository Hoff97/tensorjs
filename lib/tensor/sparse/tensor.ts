import Tensor, {
  Activation,
  DType,
  PadMode,
  TensorValues,
  tensorValuesConstructor,
} from '../../types';
import {computeStrides, getSize, indexToPos} from '../../util/shape';
import {CPUTensor} from '../cpu/tensor';

export class SparseTensor<DTpe extends DType = 'float32'> extends Tensor<DTpe> {
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
    public shape: number[],
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
    throw new Error('Method not implemented.');
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
    throw new Error('Method not implemented.');
  }
  abs(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  sin(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  cos(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  tan(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  asin(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  acos(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  atan(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  sinh(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  cosh(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  tanh(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  asinh(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  acosh(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  atanh(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  negate(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  powerScalar(power: number, factor: number): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  sigmoid(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  hardSigmoid(alpha: number, beta: number): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  sign(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  addMultiplyScalar(factor: number, add: number): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  matMul(tensor: Tensor<DTpe>): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  concat(tensor: Tensor<DTpe>, axis: number): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  clip(min?: number, max?: number): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  clipBackward(grad: Tensor<DTpe>, min?: number, max?: number): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  repeat(repeats: number[]): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  expand(shape: readonly number[]): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  copy(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  gather(axis: number, indices: CPUTensor<'uint32'>): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  setValues(values: Tensor<DTpe>, starts: number[]): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  floor(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  ceil(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
  }
  round(): Tensor<DTpe> {
    throw new Error('Method not implemented.');
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
    throw new Error('Method not implemented.');
  }
  subtract_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor<DTpe> {
    throw new Error('Method not implemented.');
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
