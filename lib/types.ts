import {CPUTensor} from './tensor/cpu/tensor';
import {compareShapes, getSize} from './util/shape';

export type PadMode = 'constant' | 'reflect' | 'edge';

export type TensorValues = Float32Array | Int32Array;

export default abstract class Tensor {
  /**
   * Gets the values of the tensor as a Float32 or Int32 Array
   *
   * @example
   * ```typescript
   * const t = new CPUTensor([2,2], [1,2,3,4]);
   * t.getValues().then(values => {
   *  //values will be a Float32Array with values [1,2,3,4]
   * });
   * ```
   */
  abstract getValues(): Promise<TensorValues>;

  /**
   * Get the shape of the tensor
   */
  abstract getShape(): ReadonlyArray<number>;

  /**
   * Constructs a tensor with the same shape and the given value everywhere
   */
  abstract constantLike(value: number): Tensor;

  /**
   * Constructs a tensor with shape [1] and the given value everywhere
   */
  abstract singleConstant(value: number): Tensor;

  /**
   * Deletes the tensor. Has the following effects depending on the backend
   * of the tensor:
   *
   * - CPU: Deletes the values
   * - WASM: Releases all memory back to the WASM runtime
   * - GPU: Releases the associated framebuffer back to the gpu memory allocator
   */
  abstract delete(): void;

  /**
   * Compares this tensor to another tensor.
   *
   * @param tensor Tensor to compare to
   * @param epsilon Optional maximum difference between the tensors. If not specified the tensors have to be exactly equal
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,2], [1,2,3,4]);
   * const b = new CPUTensor([2,2], [1.1,2.1,2.9,4.05]);
   * const c = new CPUTensor([4], [1,2,3,4]);
   * a.compare(b, 0.5).then(equal => {
   *  //equal will be true
   * });
   *
   * a.compare(b).then(equal => {
   *  //equal will be false
   * });
   *
   * a.compare(c).then(equal => {
   *  //equal will be false since the shapes of the tensors do not match
   * });
   * ```
   */
  async compare(tensor: Tensor, epsilon?: number): Promise<boolean> {
    if (!compareShapes(this.getShape(), tensor.getShape())) {
      return false;
    }

    const arrA = await this.getValues();
    const arrB = await tensor.getValues();

    if (epsilon !== undefined) {
      for (let i = 0; i < arrA.length; i += 1) {
        if (Math.abs(arrA[i] - arrB[i]) > epsilon) {
          return false;
        }
      }
    } else {
      for (let i = 0; i < arrA.length; i += 1) {
        if (arrA[i] !== arrB[i]) {
          return false;
        }
      }
    }

    return true;
  }

  protected getAxes(axes?: number | number[]) {
    let ax: number[];

    const sh = this.getShape();
    if (axes === undefined) {
      ax = [];
      for (let i = 0; i < sh.length; i++) {
        ax.push(i);
      }
    } else if (!(axes instanceof Array)) {
      ax = [axes];
    } else {
      ax = axes;
    }
    return ax;
  }

  /**
   * Sums over the specified axis/axes.
   *
   * @param axes One or multiple axes to sum over. If not specified this will sum over all axes
   * @param keepDims Wether the summation axes will be kept with size 1
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,3], [1,2,3,4,5,6]);
   *
   * a.sum(); //Will be [21]
   * a.sum(0); //Will be [5,7,9]
   * a.sum(1); //Will [6,15]
   * a.sum(0, true); //Will be [[5,7,9]]
   * ```
   */
  sum(axes?: number | number[], keepDims?: boolean): Tensor {
    const ax = this.getAxes(axes);
    keepDims = keepDims || false;
    return this.sum_impl(ax, keepDims);
  }

  /**
   * Sums over the specified axis/axes with the entries of the tensor squared.
   * This is equal to `a.multiply(a).sum(axes, keepDims)` but faster
   *
   * @param axes One or multiple axes to sum over. If not specified this will sum over all axes
   * @param keepDims Wether the summation axes will be kept with size 1
   *
   */
  sumSquare(axes?: number | number[], keepDims?: boolean): Tensor {
    const ax = this.getAxes(axes);
    keepDims = keepDims || false;
    return this.sumSquare_impl(ax, keepDims);
  }

  /**
   * Takes the product over specified axis/axes.
   *
   * @param axes One or multiple axes to take the product over. If not specified this will be all axes
   * @param keepDims Wether the product axes will be kept with size 1
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,3], [1,2,3,4,5,6]);
   *
   * a.product(); //Will be [720]
   * a.product(0); //Will be [4,10,18]
   * a.product(1); //Will [6,120]
   * a.product(0, true); //Will be [[4,10,18]]
   * ```
   */
  product(axes?: number | number[], keepDims?: boolean): Tensor {
    const ax = this.getAxes(axes);
    keepDims = keepDims || false;
    return this.product_impl(ax, keepDims);
  }

  /**
   * Takes the maximum over specified axis/axes.
   *
   * @param axes One or multiple axes to take the maximum over. If not specified this will be all axes
   * @param keepDims Wether the maximum axes will be kept with size 1
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,3], [1,2,3,4,5,6]);
   *
   * a.max(); //Will be [6]
   * a.max(0); //Will be [4,5,6]
   * a.max(1); //Will [3,6]
   * a.max(0, true); //Will be [[4,5,6]]
   * ```
   */
  max(axes?: number | number[], keepDims?: boolean): Tensor {
    const ax = this.getAxes(axes);
    keepDims = keepDims || false;
    return this.max_impl(ax, keepDims);
  }

  /**
   * Takes the minimum over specified axis/axes.
   *
   * @param axes One or multiple axes to take the minimum over. If not specified this will be all axes
   * @param keepDims Wether the minimum axes will be kept with size 1
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,3], [1,2,3,4,5,6]);
   *
   * a.min(); //Will be [1]
   * a.min(0); //Will be [1,2,3]
   * a.min(1); //Will [1,4]
   * a.min(0, true); //Will be [[1,2,3]]
   * ```
   */
  min(axes?: number | number[], keepDims?: boolean): Tensor {
    const ax = this.getAxes(axes);
    keepDims = keepDims || false;
    return this.min_impl(ax, keepDims);
  }

  /**
   * Takes the mean over the specified axis/axes.
   * This is equal to `a.sum(axes, keepDims).divide(sumSize)` (where sumSize is the number
   * of entries in the summation axes) but faster.
   *
   * @param axes One or multiple axes to take the mean over. If not specified this will take the mean over all axes
   * @param keepDims Wether the mean axes will be kept with size 1
   *
   */
  reduceMean(axes?: number | number[], keepDims?: boolean): Tensor {
    const ax = this.getAxes(axes);
    keepDims = keepDims || false;

    return this.reduceMean_impl(ax, keepDims);
  }

  /**
   * Takes the log of the sum over the specified axis
   * This is equal to `a.sum(axes, keepDims).log()` (where sumSize is the number
   * of entries in the summation axes) but faster.
   *
   * @param axes One or multiple axes to take the mean over. If not specified this will take the mean over all axes
   * @param keepDims Wether the mean axes will be kept with size 1
   *
   */
  reduceLogSum(axes?: number | number[], keepDims?: boolean): Tensor {
    const ax = this.getAxes(axes);
    keepDims = keepDims || false;

    return this.reduceLogSum_impl(ax.sort(), keepDims);
  }

  /**
   * Takes the log of the sum over the exp of the specified axis
   * This is equal to `a.sum(axes, keepDims).log()` (where sumSize is the number
   * of entries in the summation axes) but faster.
   *
   * @param axes One or multiple axes to take the mean over. If not specified this will take the mean over all axes
   * @param keepDims Wether the mean axes will be kept with size 1
   *
   */
  reduceLogSumExp(axes?: number | number[], keepDims?: boolean): Tensor {
    const ax = this.getAxes(axes);
    keepDims = keepDims || false;

    return this.reduceLogSumExp_impl(ax, keepDims);
  }

  /**
   * Takes the mean over the specified axis/axes with the entries of the tensor squared.
   * This is equal to `a.multiply(a).sum(axes, keepDims).divide(sumSize)` (where sumSize is the number
   * of entries in the summation axes) but faster.
   *
   * @param axes One or multiple axes to take the mean over. If not specified this will take the mean over all axes
   * @param keepDims Wether the mean axes will be kept with size 1
   *
   */
  reduceMeanSquare(axes?: number | number[], keepDims?: boolean): Tensor {
    const ax = this.getAxes(axes);
    keepDims = keepDims || false;

    return this.reduceMeanSquare_impl(ax, keepDims);
  }

  /**
   * Convolves this tensor with the specified kernel.
   *
   * This tensor should have shape [N,C,D1,D2,...] where D1,D2,... are the spatial dimensions.
   *
   * Behaves according to https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
   *
   * @param kernel Convolution kernel with shape [M,C/G,K1,K2] where G is the group parameter
   * @param bias Optional bias to add to the result with shape [M]
   * @param dilations Per axis dilations for the spatial dimension. Defaults to 1 for all axes
   * @param group Group parameter
   * @param pads Padding to add to the input for each spatial dimension. Defaults to 0 for all axes
   * @param strides Convolution stride for each spatial dimension. Defaults to 1 for all axes
   * @param activation Optional activation to apply. Defaults to the identity (so no activation)
   */
  conv(
    kernel: Tensor,
    bias?: Tensor,
    dilations?: number[],
    group?: number,
    pads?: number[],
    strides?: number[],
    activation?: Activation
  ): Tensor {
    const sh = this.getShape();
    const dataRank = sh.length - 2;

    dilations = dilations || new Array(dataRank).fill(1);
    group = group || 1;
    pads = pads || new Array(dataRank * 2).fill(0);
    strides = strides || new Array(dataRank).fill(1);

    if (activation === undefined) {
      activation = 'id';
    }

    return this.conv_impl(
      kernel,
      dilations,
      group,
      pads,
      strides,
      activation,
      bias
    );
  }

  /**
   * Calculates the transpose convolution
   *
   * This tensor should have shape [N,C,D1,D2,...] where D1,D2,... are the spatial dimensions.
   *
   * @param kernel Convolution kernel with shape [M,C/G,K1,K2] where G is the group parameter
   * @param dilations Per axis dilations for the spatial dimension. Defaults to 1 for all axes
   * @param group Group parameter
   * @param pads Padding to add to the input for each spatial dimension. Defaults to 0 for all axes
   * @param strides Convolution stride for each spatial dimension. Defaults to 1 for all axes
   */
  convTranspose(
    kernel: Tensor,
    dilations?: number[],
    group?: number,
    pads?: number[],
    strides?: number[]
  ): Tensor {
    const sh = this.getShape();
    const dataRank = sh.length - 2;

    dilations = dilations || new Array(dataRank).fill(1);
    group = group || 1;
    pads = pads || new Array(dataRank * 2).fill(0);
    strides = strides || new Array(dataRank).fill(1);

    return this.convTranspose_impl(kernel, dilations, group, pads, strides);
  }

  /**
   * Pads the input according to the padding mode. The input has shape [D1,D2,..]
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,2],[1,2,3,4]);
   * a.pad([1,1,1,1],'constant',5);
   * //Result will be:
   * // [[5,5,5,5],
   * //  [5,1,2,5],
   * //  [5,3,4,5],
   * //  [5,5,5,5]]
   * a.pad([1,1,1,1],'edge');
   * //Result will be:
   * // [[1,1,2,2],
   * //  [1,1,2,2],
   * //  [3,3,4,4],
   * //  [3,3,4,4]]
   *
   * a.pad([2,2,2,2],'reflect');
   * //Result will be:
   * // [[4,3,3,4,4,3],
   * //  [2,1,1,2,2,1],
   * //  [2,1,1,2,2,1],
   * //  [4,3,3,4,4,3],
   * //  [4,3,3,4,4,3],
   * //  [2,1,1,2,2,1]]
   * ```
   *
   * @param pads Padding size of each input. Specified as [startpad_D1,startpad_D2,...,startpad_DN,endpad_D1,endpad_D2,...]
   * @param mode Padding mode. One of 'constant', 'edge', 'reflect'. Defaults to 'constant'
   * @param value Value for constant padding. Defaults to 0.0
   */
  pad(pads: number[], mode?: PadMode, value?: number): Tensor {
    if (mode === undefined) {
      mode = 'constant';
    }
    if (value === undefined) {
      value = 0;
    }
    return this.pad_impl(pads, mode, value);
  }

  /**
   * Performs average pooling over the spatial dimensions of this tensor with
   * shape [N,C,D1,D2,..]
   * @param kernelShape Size of the average pooling dimension
   * @param pads Padding of the input specified as [startpad_D1,startpad_D2,...,startpad_DN,endpad_D1,endpad_D2,...]
   *             Padding value will be 0. Defaults to 0 for all axes
   * @param strides Stride size of the average pooling kernel. Defaults to 1 for all axes
   * @param includePad Wether padded values should be included in the average (or masked out). Defaults to false
   */
  averagePool(
    kernelShape: number[],
    pads?: number[],
    strides?: number[],
    includePad?: boolean
  ): Tensor {
    const sh = this.getShape();
    const dataRank = sh.length - 2;

    pads = pads || new Array(dataRank * 2).fill(0);
    strides = strides || new Array(dataRank).fill(1);
    includePad = includePad || false;

    return this.averagePool_impl(kernelShape, pads, strides, includePad);
  }

  /**
   * Reshape the tensor to the specified shape
   *
   * At most one value in the shape can be -1, which will be replaced by the inferred size for this dimension.
   *
   * @param shape New shape of the tensor
   * @param copy Wether the tensor values should be copied. Only has an effect on GPU tensors
   */
  reshape(shape: readonly number[], copy?: boolean): Tensor {
    let shSize = 1;
    let negIndex = -1;
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] === -1) {
        negIndex = i;
      } else {
        shSize *= shape[i];
      }
    }

    if (copy === undefined) {
      copy = true;
    }

    if (negIndex !== -1) {
      const currShape = this.getShape();
      const currSize = getSize(currShape);
      const _shape = [...shape];

      _shape[negIndex] = currSize / shSize;

      return this.reshape_impl(_shape, copy);
    }
    return this.reshape_impl(shape, copy);
  }

  protected abstract reshape_impl(
    shape: readonly number[],
    copy: boolean
  ): Tensor;

  /**
   * Takes the exponential of each value of the tensor
   */
  abstract exp(): Tensor;

  /**
   * Takes the natural logarithm of each value of the tensor
   */
  abstract log(): Tensor;

  /**
   * Takes the square root of each value of the tensor
   */
  abstract sqrt(): Tensor;

  /**
   * Takes the absolute of each value of the tensor
   */
  abstract abs(): Tensor;

  /**
   * Takes the sinus of each value of the tensor
   */
  abstract sin(): Tensor;

  /**
   * Takes the cosine of each value of the tensor
   */
  abstract cos(): Tensor;

  /**
   * Takes the tangens of each value of the tensor
   */
  abstract tan(): Tensor;

  /**
   * Negates all entries of the tensor
   */
  abstract negate(): Tensor;

  /**
   * Computes the element wise sigmoid of all values
   */
  abstract sigmoid(): Tensor;

  /**
   * Computes the value-wise sign which is:
   *  - (-1) if x < 0
   *  - 1 otherwise
   */
  abstract sign(): Tensor;

  alignShapes(
    shape1: readonly number[],
    shape2: readonly number[]
  ): (readonly number[])[] {
    if (compareShapes(shape1, shape2)) {
      return [shape1, shape2, shape1];
    }
    if (shape1.length < shape2.length) {
      shape1 = [...shape1];
      const prepend = shape2.length - shape1.length;
      (shape1 as number[]).unshift(...new Array(prepend).fill(1));
    } else if (shape2.length < shape1.length) {
      shape2 = [...shape2];
      const prepend = shape1.length - shape2.length;
      (shape2 as number[]).unshift(...new Array(prepend).fill(1));
    }

    const resultShape = new Array(shape1.length).fill(1);
    for (let i = 0; i < shape1.length; i++) {
      resultShape[i] = Math.max(shape1[i], shape2[i]);
    }

    return [shape1, shape2, resultShape];
  }

  /**
   * Align the shapes of this tensor and the given tensor according to
   * the broadcasting rules:
   * https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
   *
   * @param tensor Tensor of which the shapes should be aligned
   */
  alignTensor(tensor: Tensor) {
    let thisShape = this.getShape();
    let thatShape = tensor.getShape();
    if (compareShapes(thisShape, thatShape)) {
      return [this, tensor, thisShape];
    }
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    let th: Tensor = this;
    if (thisShape.length < thatShape.length) {
      thisShape = [...thisShape];
      const prepend = thatShape.length - thisShape.length;
      (thisShape as number[]).unshift(...new Array(prepend).fill(1));
      th = this.reshape(thisShape, false);
    } else if (thatShape.length < thisShape.length) {
      thatShape = [...thatShape];
      const prepend = thisShape.length - thatShape.length;
      (thatShape as number[]).unshift(...new Array(prepend).fill(1));
      tensor = tensor.reshape(thatShape, false);
    }

    const resultShape = new Array(thisShape.length).fill(1);
    for (let i = 0; i < thisShape.length; i++) {
      resultShape[i] = Math.max(thisShape[i], thatShape[i]);
    }
    return [th, tensor, resultShape];
  }

  /**
   * Adds two tensors. Supports broadcasting
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,2],[1,2,3,4]);
   * const b = new CPUTensor([2,2],[5,6,7,8]);
   * const c = new CPUTensor([1],[2]);
   *
   * a.add(b);
   * //Will be
   * // [[6,8],
   * //  [10,12]]
   *
   * a.add(c);
   * //Will be
   * // [[3,4],
   * //  [5,6]]
   * ```
   */
  add(tensor: Tensor, alpha?: number, beta?: number) {
    if (alpha === undefined) {
      alpha = 1;
    }
    if (beta === undefined) {
      beta = 1;
    }
    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.add_impl(
      th as Tensor,
      tens as Tensor,
      resultShape as number[],
      alpha,
      beta
    );
  }

  /**
   * Subtracts two tensors. Supports broadcasting
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,2],[5,6,7,8]);
   * const b = new CPUTensor([2,2],[1,2,3,4]);
   * const c = new CPUTensor([1],[2]);
   *
   * a.subtract(b);
   * //Will be
   * // [[4,4],
   * //  [4,4]]
   *
   * a.subtract(c);
   * //Will be
   * // [[3,4],
   * //  [5,6]]
   * ```
   */
  subtract(tensor: Tensor, alpha?: number, beta?: number) {
    if (alpha === undefined) {
      alpha = 1;
    }
    if (beta === undefined) {
      beta = 1;
    }

    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.subtract_impl(
      th as Tensor,
      tens as Tensor,
      resultShape as number[],
      alpha,
      beta
    );
  }

  /**
   * Multiplies two tensors. Supports broadcasting
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,2],[1,2,3,4]);
   * const b = new CPUTensor([2,2],[5,6,7,8]);
   * const c = new CPUTensor([1],[2]);
   *
   * a.multiply(b);
   * //Will be
   * // [[5,12],
   * //  [21,32]]
   *
   * a.multiply(c);
   * //Will be
   * // [[2,4]
   *     [6,8]]
   * ```
   */
  multiply(tensor: Tensor, alpha?: number) {
    if (alpha === undefined) {
      alpha = 1;
    }
    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.multiply_impl(
      th as Tensor,
      tens as Tensor,
      resultShape as number[],
      alpha
    );
  }

  multiplyScalar(value: number) {
    return this.addMultiplyScalar(value, 0);
  }

  addScalar(value: number) {
    return this.addMultiplyScalar(1, value);
  }

  abstract addMultiplyScalar(factor: number, add: number): Tensor;

  /**
   * Divides two tensors. Supports broadcasting
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,2],[5,6,7,8]);
   * const b = new CPUTensor([2,2],[1,2,3,4]);
   * const c = new CPUTensor([1],[2]);
   *
   * a.divide(b);
   * //Will be
   * // [[5,3],
   * //  [2.333,2]]
   *
   * a.divide(c);
   * //Will be
   * // [[2.5,3],
   * //  [3.5,4]]
   * ```
   */
  divide(tensor: Tensor, alpha?: number) {
    if (alpha === undefined) {
      alpha = 1;
    }

    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.divide_impl(
      th as Tensor,
      tens as Tensor,
      resultShape as number[],
      alpha
    );
  }

  /**
   * Takes the positionwise power. Supports broadcasting
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,2],[5,6,7,8]);
   * const b = new CPUTensor([2,2],[2,3,2,3]);
   * const c = new CPUTensor([1],[2]);
   *
   * a.power(b);
   * //Will be
   * // [[25,216],
   * //  [49,512]]
   *
   * a.power(c);
   * //Will be
   * // [[25,36],
   * //  [49,64]]
   * ```
   */
  power(tensor: Tensor) {
    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.power_impl(
      th as Tensor,
      tens as Tensor,
      resultShape as number[]
    );
  }

  /**
   * Transposes the tensor according to the given permutation
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,2],[5,6,7,8]);
   *
   * a.transpose();
   * //Will be
   * // [[5,7],
   * //  [6,8]]
   * ```
   * @param permutation Permutation for the axes. Default is the reverse axis order
   */
  transpose(permutation?: number[]): Tensor {
    if (permutation === undefined) {
      const shape = this.getShape();
      const rank = shape.length;
      permutation = [];
      for (let i = 0; i < rank; i++) {
        permutation.push(rank - i - 1);
      }
    }
    return this.transpose_impl(permutation);
  }

  /**
   * Takes the softmax along the given axis
   * https://en.wikipedia.org/wiki/Softmax_function
   */
  softmax(axis: number) {
    const max = this.max(axis, true);
    const normalized = this.subtract(max);
    const exp = normalized.exp();

    const sum = exp.sum(axis, true);
    const result = exp.divide(sum);

    max.delete();
    normalized.delete();
    exp.delete();
    sum.delete();

    return result;
  }

  /**
   * Calculates the general matrix product.
   * https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
   *
   * A and B can have batch dimensions. Their last two dimensions should
   * correspond to the dimensions for the matrix product
   *
   * @param b Second matrix for the matrix product
   * @param aTranspose If the last two dimensions of a are transposed. Defaults to false
   * @param bTranspose If the last two dimensions of a are transposed. Defaults to false
   * @param alpha Alpha parameter. Defaults to 1.0
   * @param c Optional tensor to add to the result.
   * @param beta Beta parameter, only used if c is specified. Defaults to 1.0
   */
  gemm(
    b: Tensor,
    aTranspose?: boolean,
    bTranspose?: boolean,
    alpha?: number,
    c?: Tensor,
    beta?: number
  ): Tensor {
    aTranspose = aTranspose || false;
    bTranspose = bTranspose || false;
    alpha = alpha !== undefined ? alpha : 1;
    beta = beta !== undefined ? beta : 1;

    if (c !== undefined) {
      const aShape = this.getShape();
      let cShape = c.getShape();
      const aRank = aShape.length;
      const cRank = cShape.length;

      if (aRank > cRank) {
        cShape = [...new Array(aRank - cRank).fill(1), ...cShape];
        c = c.reshape(cShape, false);
      }
    }

    return this.gemm_impl(b, aTranspose, bTranspose, alpha, beta, c);
  }

  /**
   * Takes a slice of the tensor along the specified axes.
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,2],[5,6,7,8]);
   *
   * a.slice([0],[1],[0]);
   * //Will be
   * // [[5,6]]
   *
   * a.slice([0],[1],[1]);
   * //Will be
   * // [[5],
   *     [6]]
   * ```
   *
   * @param starts Start of the slice for each axis
   * @param ends End of the slice for each axis - Exclusive (the end index will not be included in the slice)
   * @param axes Axes to slice. Defaults to all axes
   */
  slice(starts: number[], ends: number[], axes?: number[]): Tensor {
    const shape = this.getShape();
    const rank = shape.length;
    if (axes === undefined) {
      axes = [];
      for (let i = 0; i < rank; i++) {
        axes.push(i);
      }
    }
    starts = [...starts];
    ends = [...ends];
    for (let i = 0; i < axes.length; i++) {
      const sh = shape[axes[i]];
      if (starts[i] < 0) {
        starts[i] += sh;
      } else if (starts[i] >= sh) {
        starts[i] = sh;
      }
      if (ends[i] < 0) {
        ends[i] += sh;
      } else if (ends[i] >= sh) {
        ends[i] = sh;
      }
    }
    return this.slice_impl(starts, ends, axes);
  }

  /**
   * Calculates the matrix product. This tensor should have shape [M,N]
   *
   * @param tensor Matrix to multiply with. Should have shape [N,O]
   *
   * @result Tensor with shape [M,O]
   */
  abstract matMul(tensor: Tensor): Tensor;

  /**
   * Concatenate the two tensors along the given axis
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,2],[1,2,3,4]);
   * const b = new CPUTensor([2,2],[5,6,7,8]);
   *
   * a.concat(b,0);
   * //Will be
   * // [[1,2],
   * //  [3,4],
   * //  [5,6],
   * //  [7,8]]
   *
   * a.concat(b,1);
   * //Will be
   * // [[1,2,5,6],
   * //  [3,4,7,8],
   * ```
   */
  abstract concat(tensor: Tensor, axis: number): Tensor;

  /**
   * Clips the tensor values between the minimum and maximum
   * @param min Minimum value. Defaults to the minimum possible value
   * @param max Maximum value. Defaults to the maximum possible value
   */
  abstract clip(min?: number, max?: number): Tensor;

  /**
   * Backward pass for clip
   * @param grad Gradient from which values should be selected
   * @param min Minimum value. Defaults to the minimum possible value
   * @param max Maximum value. Defaults to the maximum possible value
   */
  abstract clipBackward(grad: Tensor, min?: number, max?: number): Tensor;

  /**
   * Repeat the tensor along each dimension
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,2],[1,2,3,4]);
   *
   * a.repeat([2,1]);
   * //Will be
   * // [[1,2],
   * //  [3,4],
   * //  [1,2],
   * //  [3,4]]
   *
   * a.repeat([1,2]);
   * //Will be
   * // [[1,2,1,2],
   * //  [3,4,3,4],
   * ```
   *
   * @param repeats Number of repetitions along each dimension
   */
  abstract repeat(repeats: number[]): Tensor;

  abstract expand(shape: readonly number[]): Tensor;

  /**
   * Copy the tensor.
   * If the tensor is a GPU tensor, you can specify a precision (16/32)
   */
  abstract copy(): Tensor;

  /**
   * Gather values along the given axis, according to the indices
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,2],[1,2,3,4]);
   * const indices = new CPUTensor([2],[0,0])
   *
   * a.gather(0,indices);
   * //Will be
   * // [[1,2],
   * //  [1,2]]
   *
   * a.gather(1,indices);
   * //Will be
   * // [[1,1],
   * //  [3,3]]
   * ```
   */
  abstract gather(axis: number, indices: CPUTensor): Tensor;

  /**
   * Sets the values in the current tensor to the given values.
   * Starts at the specified start indices at each axis.
   * Note that this will not occur in place, but will generate
   * a new tensor.
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,2],[1,2,3,4]);
   * const b = new CPUTensor([1,2],[5,6])
   *
   * a.set(b,[1,0]);
   * //Will be
   * // [[1,2],
   * //  [5,6]]
   * ```
   */
  abstract setValues(values: Tensor, starts: number[]): Tensor;

  /**
   * Rounds each tensor value to the nearest upper integer
   */
  abstract floor(): Tensor;

  /**
   * Rounds each tensor value to the nearest lower integer
   */
  abstract ceil(): Tensor;

  /**
   * Scales the tensor up/down according to the specified scales.
   * Uses nearest neighbor sampling
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,2],[1,2,3,4]);
   *
   * a.upsample([2,2]);
   * //Will be
   * // [[1,1,2,2],
   * //  [1,1,2,2],
   * //  [3,3,4,4],
   * //  [3,3,4,4]]
   * ```
   */
  abstract upsample(scales: number[]): Tensor;

  /**
   * Normalizes the tensor according to the following formula:
   * ```
   * x' = (x-mean)/sqrt(variance + epsilon)
   * x'' = x'*scale + bias
   * ```
   */
  abstract normalize(
    mean: Tensor,
    variance: Tensor,
    epsilon: number,
    scale: Tensor,
    bias: Tensor
  ): Tensor;

  abstract add_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor;

  abstract subtract_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor;

  abstract multiply_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[],
    alpha: number
  ): Tensor;

  abstract divide_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[],
    alpha: number
  ): Tensor;

  abstract power_impl(
    th: Tensor,
    tensor: Tensor,
    resultShape: readonly number[]
  ): Tensor;

  abstract gemm_impl(
    b: Tensor,
    aTranspose: boolean,
    bTranspose: boolean,
    alpha: number,
    beta: number,
    C?: Tensor
  ): Tensor;

  protected abstract sum_impl(axes: number[], keepDims: boolean): Tensor;
  protected abstract sumSquare_impl(axes: number[], keepDims: boolean): Tensor;

  protected abstract product_impl(axes: number[], keepDims: boolean): Tensor;

  protected abstract max_impl(axes: number[], keepDims: boolean): Tensor;

  protected abstract min_impl(axes: number[], keepDims: boolean): Tensor;

  protected abstract reduceMean_impl(axes: number[], keepDims: boolean): Tensor;

  protected abstract reduceMeanSquare_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor;

  protected abstract reduceLogSum_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor;

  protected abstract reduceLogSumExp_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor;

  protected abstract conv_impl(
    kernel: Tensor,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[],
    activation: Activation,
    bias?: Tensor
  ): Tensor;

  protected abstract convTranspose_impl(
    kernel: Tensor,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[]
  ): Tensor;

  protected abstract pad_impl(
    pads: number[],
    mode: PadMode,
    value: number
  ): Tensor;

  protected abstract averagePool_impl(
    kernelShape: number[],
    pads: number[],
    strides: number[],
    includePad: boolean
  ): Tensor;

  protected abstract transpose_impl(permutation: number[]): Tensor;

  protected abstract slice_impl(
    starts: number[],
    ends: number[],
    axes: number[]
  ): Tensor;
}

export type Activation = 'id' | 'relu' | 'relu6';

export type Precision = 16 | 32;
