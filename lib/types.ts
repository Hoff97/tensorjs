import {CPUTensor} from './tensor/cpu/tensor';
import {compareShapes, getSize} from './util/shape';

export type PadMode = 'constant' | 'reflect' | 'edge';

/**
 * Type that is returned when calling `tensor.getValues()`
 */
export type TensorValues = {
  float64: Float64Array;
  float32: Float32Array;
  float16: Float32Array;
  int32: Int32Array;
  int16: Int16Array;
  int8: Int8Array;
  uint32: Uint32Array;
  uint16: Uint16Array;
  uint8: Uint8Array;
};

export const tensorValuesConstructor = {
  float64: Float64Array,
  float32: Float32Array,
  float16: Float32Array,
  int32: Int32Array,
  int16: Int16Array,
  int8: Int8Array,
  uint32: Uint32Array,
  uint16: Uint16Array,
  uint8: Uint8Array,
};

/**
 * Tensor data types available in tensor-js
 */
export type DType =
  | 'float64'
  | 'float32'
  | 'float16'
  | 'int32'
  | 'int16'
  | 'int8'
  | 'uint32'
  | 'uint16'
  | 'uint8';

/**
 * Activation functions supported in some operators
 */
export type Activation = 'id' | 'relu' | 'relu6';

/**
 * Multi-dimensional array ala numpy.
 *
 * A tensor is any multidimensional array. The number of
 * dimensions is called the rank, and the size of all dimensions the shape.
 *
 * @example
 * ```typescript
 * const a = [[1,2,3],[4,5,6]];
 * ```
 * here a has rank 2 and shape [2,3].
 *
 * Tensors store values of a particular data type like floats or integers.
 * The datatype can be accessed via the dtype property.
 *
 * Many operations can be done on tensors. For fast execution, three different
 * backends exist:
 * - CPU: Simple to use and works in any browser, but not particularly fast
 * - WebAssembly: Reasonably fast and works in most modern browsers
 * - WebGL: Very fast when a GPU is available, but comes with some restrictions
 */
export default abstract class Tensor<DTpe extends DType = 'float32'> {
  /**
   * Data type of the tensor
   */
  public dtype: DTpe;

  constructor(dtype: DTpe) {
    this.dtype = dtype;
  }
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
  abstract getValues(): Promise<TensorValues[DTpe]>;

  /**
   * Get the shape of the tensor
   */
  abstract getShape(): ReadonlyArray<number>;

  /**
   * Constructs a tensor with the same shape and the given value everywhere
   */
  abstract constantLike(value: number): Tensor<DTpe>;

  /**
   * Constructs a tensor with shape [1] and the given value everywhere
   */
  abstract singleConstant(value: number): Tensor<DTpe>;

  abstract cast<DTpe2 extends DType>(dtype: DTpe2): Tensor<DTpe2>;

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
  async compare(tensor: Tensor<DTpe>, epsilon?: number): Promise<boolean> {
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
      for (let i = 0; i < ax.length; i++) {
        if (ax[i] < 0) {
          ax[i] += this.getShape().length;
        }
      }
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
  sum(axes?: number | number[], keepDims?: boolean): Tensor<DTpe> {
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
  sumSquare(axes?: number | number[], keepDims?: boolean): Tensor<DTpe> {
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
  product(axes?: number | number[], keepDims?: boolean): Tensor<DTpe> {
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
  max(axes?: number | number[], keepDims?: boolean): Tensor<DTpe> {
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
  min(axes?: number | number[], keepDims?: boolean): Tensor<DTpe> {
    const ax = this.getAxes(axes);
    keepDims = keepDims || false;
    return this.min_impl(ax, keepDims);
  }

  /**
   * Gives the indices of the maximum over the specifed axes.
   *
   * @param axes One or multiple axes to take the maximum over. If not specified this will be all axes
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,3], [1,2,3,4,5,6]);
   *
   * a.argMax(); //Will be [[1,2]]
   * a.argMax(0); //Will be [1,1,1] (since the maximum values 4,5,6 are located in the second row)
   * a.argMax(1); //Will [2,2] (Since the maximum values 3,6 are located in the third column)
   * ```
   */
  argMax(axes?: number | number[], selectLast?: boolean): Tensor<'uint32'> {
    const ax = this.getAxes(axes);
    if (selectLast === undefined) {
      selectLast = false;
    }
    return this.argMax_impl(ax, selectLast);
  }

  /**
   * Gives the indices of the minimum over the specifed axes.
   *
   * @param axes One or multiple axes to take the minimum over. If not specified this will be all axes
   *
   * @example
   * ```typescript
   * const a = new CPUTensor([2,3], [1,2,3,4,5,6]);
   *
   * a.argMin(); //Will be [[0,0]]
   * a.argMin(0); //Will be [0,0,0] (since the minimum values 1,2,3 are located in the first row)
   * a.argMin(1); //Will [0,0] (Since the minimum values 1,4 are located in the first column)
   * ```
   */
  argMin(axes?: number | number[], selectLast?: boolean): Tensor<'uint32'> {
    const ax = this.getAxes(axes);
    if (selectLast === undefined) {
      selectLast = false;
    }
    return this.argMin_impl(ax, selectLast);
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
  reduceMean(axes?: number | number[], keepDims?: boolean): Tensor<DTpe> {
    const ax = this.getAxes(axes);
    keepDims = keepDims || false;

    return this.reduceMean_impl(ax, keepDims);
  }

  /**
   * Takes the log of the sum over the specified axis
   * This is equal to `a.sum(axes, keepDims).log()` (where sumSize is the number
   * of entries in the summation axes) but faster.
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   *
   * @param axes One or multiple axes to take the mean over. If not specified this will take the mean over all axes
   * @param keepDims Wether the mean axes will be kept with size 1
   *
   */
  reduceLogSum(axes?: number | number[], keepDims?: boolean): Tensor<DTpe> {
    const ax = this.getAxes(axes);
    keepDims = keepDims || false;

    return this.reduceLogSum_impl(ax.sort(), keepDims);
  }

  /**
   * Takes the log of the sum over the exp of the specified axis
   * This is equal to `a.sum(axes, keepDims).log()` (where sumSize is the number
   * of entries in the summation axes) but faster.
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   *
   * @param axes One or multiple axes to take the mean over. If not specified this will take the mean over all axes
   * @param keepDims Wether the mean axes will be kept with size 1
   *
   */
  reduceLogSumExp(axes?: number | number[], keepDims?: boolean): Tensor<DTpe> {
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
  reduceMeanSquare(axes?: number | number[], keepDims?: boolean): Tensor<DTpe> {
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
    kernel: Tensor<DTpe>,
    bias?: Tensor<DTpe>,
    dilations?: number[],
    group?: number,
    pads?: number[],
    strides?: number[],
    activation?: Activation
  ): Tensor<DTpe> {
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
    kernel: Tensor<DTpe>,
    dilations?: number[],
    group?: number,
    pads?: number[],
    strides?: number[]
  ): Tensor<DTpe> {
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
  pad(pads: number[], mode?: PadMode, value?: number): Tensor<DTpe> {
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
  ): Tensor<DTpe> {
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
  reshape(shape: readonly number[], copy?: boolean): Tensor<DTpe> {
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
  ): Tensor<DTpe>;

  /**
   * Takes the exponential of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract exp(): Tensor<DTpe>;

  /**
   * Takes the natural logarithm of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract log(): Tensor<DTpe>;

  /**
   * Takes the square root of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract sqrt(): Tensor<DTpe>;

  /**
   * Takes the absolute of each value of the tensor
   *
   * Note that this can only be called on tensors with a signed data type (float*, int32, int16, int8)
   */
  abstract abs(): Tensor<DTpe>;

  /**
   * Takes the sinus of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract sin(): Tensor<DTpe>;

  /**
   * Takes the cosine of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract cos(): Tensor<DTpe>;

  /**
   * Takes the tangens of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract tan(): Tensor<DTpe>;

  /**
   * Takes the arcus sinus of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract asin(): Tensor<DTpe>;

  /**
   * Takes the arcus cosine of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract acos(): Tensor<DTpe>;

  /**
   * Takes the arcus tangens of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract atan(): Tensor<DTpe>;

  /**
   * Takes the hyperbolic sinus of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract sinh(): Tensor<DTpe>;

  /**
   * Takes the hyperbolic cosine of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract cosh(): Tensor<DTpe>;

  /**
   * Takes the hyperbolic tangens of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract tanh(): Tensor<DTpe>;

  /**
   * Takes the inverse hyperbolic sinus of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract asinh(): Tensor<DTpe>;

  /**
   * Takes the inverse hyperbolic cosine of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract acosh(): Tensor<DTpe>;

  /**
   * Takes the inverse hyperbolic tangens of each value of the tensor
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract atanh(): Tensor<DTpe>;

  /**
   * Negates all entries of the tensor
   *
   * Note that this can only be called on tensors with a signed data type (float*, int32, int16, int8)
   */
  abstract negate(): Tensor<DTpe>;

  /**
   * Takes element wise power and multiplies with the given factor
   */
  abstract powerScalar(power: number, factor: number): Tensor<DTpe>;

  /**
   * Computes the element wise sigmoid of all values
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract sigmoid(): Tensor<DTpe>;

  /**
   * Computes the element wise hard sigmoid of all values given
   * by `y = max(0, min(1, alpha * x + beta))`
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract hardSigmoid(alpha: number, beta: number): Tensor<DTpe>;

  /**
   * Computes the value-wise sign which is:
   *  - (-1) if x < 0
   *  - 0 if x == 0
   *  - 1 otherwise
   *
   * Note that this can only be called on tensors with a signed data type (float*, int32, int16, int8)
   */
  abstract sign(): Tensor<DTpe>;

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
  alignTensor(tensor: Tensor<DTpe>) {
    let thisShape = this.getShape();
    let thatShape = tensor.getShape();
    if (compareShapes(thisShape, thatShape)) {
      return [this, tensor, thisShape];
    }
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    let th: Tensor<DTpe> = this;
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
  add(tensor: Tensor<DTpe>, alpha?: number, beta?: number) {
    if (alpha === undefined) {
      alpha = 1;
    }
    if (beta === undefined) {
      beta = 1;
    }
    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.add_impl(
      th as Tensor<DTpe>,
      tens as Tensor<DTpe>,
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
  subtract(tensor: Tensor<DTpe>, alpha?: number, beta?: number) {
    if (alpha === undefined) {
      alpha = 1;
    }
    if (beta === undefined) {
      beta = 1;
    }

    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.subtract_impl(
      th as Tensor<DTpe>,
      tens as Tensor<DTpe>,
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
  multiply(tensor: Tensor<DTpe>, alpha?: number) {
    if (alpha === undefined) {
      alpha = 1;
    }
    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.multiply_impl(
      th as Tensor<DTpe>,
      tens as Tensor<DTpe>,
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

  abstract addMultiplyScalar(factor: number, add: number): Tensor<DTpe>;

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
  divide(tensor: Tensor<DTpe>, alpha?: number) {
    if (alpha === undefined) {
      alpha = 1;
    }

    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.divide_impl(
      th as Tensor<DTpe>,
      tens as Tensor<DTpe>,
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
  power(tensor: Tensor<DTpe>) {
    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.power_impl(
      th as Tensor<DTpe>,
      tens as Tensor<DTpe>,
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
  transpose(permutation?: number[]): Tensor<DTpe> {
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
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
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
    b: Tensor<DTpe>,
    aTranspose?: boolean,
    bTranspose?: boolean,
    alpha?: number,
    c?: Tensor<DTpe>,
    beta?: number
  ): Tensor<DTpe> {
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
  slice(
    starts: number[],
    ends: number[],
    axes?: number[],
    steps?: number[]
  ): Tensor<DTpe> {
    const shape = this.getShape();
    const rank = shape.length;
    if (axes === undefined) {
      axes = [];
      for (let i = 0; i < rank; i++) {
        axes.push(i);
      }
    } else {
      axes = axes.map(x => (x < 0 ? x + rank : x));
    }
    if (steps === undefined) {
      steps = new Array(rank).fill(1);
    }
    starts = [...starts];
    ends = [...ends];
    for (let i = 0; i < axes.length; i++) {
      const sh = shape[axes[i]];
      if (starts[i] < 0) {
        starts[i] += sh;
      } else if (starts[i] >= sh) {
        if (steps[i] > 0) {
          starts[i] = sh;
        } else {
          starts[i] = sh - 1;
        }
      }
      if (ends[i] < 0) {
        ends[i] += sh;
      } else if (ends[i] >= sh) {
        ends[i] = sh;
      }
    }
    return this.slice_impl(starts, ends, axes, steps);
  }

  /**
   * Calculates the matrix product. This tensor should have shape [M,N]
   *
   * @param tensor Matrix to multiply with. Should have shape [N,O]
   *
   * @result Tensor with shape [M,O]
   */
  abstract matMul(tensor: Tensor<DTpe>): Tensor<DTpe>;

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
  abstract concat(tensor: Tensor<DTpe>, axis: number): Tensor<DTpe>;

  /**
   * Clips the tensor values between the minimum and maximum
   * @param min Minimum value. Defaults to the minimum possible value
   * @param max Maximum value. Defaults to the maximum possible value
   */
  abstract clip(min?: number, max?: number): Tensor<DTpe>;

  /**
   * Backward pass for clip
   * @param grad Gradient from which values should be selected
   * @param min Minimum value. Defaults to the minimum possible value
   * @param max Maximum value. Defaults to the maximum possible value
   */
  abstract clipBackward(
    grad: Tensor<DTpe>,
    min?: number,
    max?: number
  ): Tensor<DTpe>;

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
  abstract repeat(repeats: number[]): Tensor<DTpe>;

  abstract expand(shape: readonly number[]): Tensor<DTpe>;

  squeeze(): Tensor<DTpe> {
    const sh = this.getShape();
    const newShape = [];
    for (const a of sh) {
      if (a !== 1) {
        newShape.push(a);
      }
    }
    return this.reshape(newShape);
  }

  flatten(axis?: number): Tensor<DTpe> {
    if (axis === undefined) {
      axis = 1;
    }
    const sh = this.getShape();
    if (axis < 0) {
      axis += sh.length;
    }
    const newShape = [
      getSize(sh.slice(0, axis), 1),
      getSize(sh.slice(axis), 1),
    ];
    return this.reshape(newShape);
  }

  /**
   * Copy the tensor.
   * If the tensor is a GPU tensor, you can specify a precision (16/32)
   */
  abstract copy(): Tensor<DTpe>;

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
  abstract gather(axis: number, indices: CPUTensor<'uint32'>): Tensor<DTpe>;

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
  abstract setValues(values: Tensor<DTpe>, starts: number[]): Tensor<DTpe>;

  /**
   * Rounds each tensor value to the nearest lower integer
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract floor(): Tensor<DTpe>;

  /**
   * Rounds each tensor value to the nearest upper integer
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract ceil(): Tensor<DTpe>;

  /**
   * Rounds each tensor value to the nearest integer.
   * When the value is 0.5 it rounds up.
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract round(): Tensor<DTpe>;

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
  abstract upsample(scales: number[]): Tensor<DTpe>;

  /**
   * Normalizes the tensor according to the following formula:
   * ```
   * x' = (x-mean)/sqrt(variance + epsilon)
   * x'' = x'*scale + bias
   * ```
   *
   * Note that this can only be called on tensors with a float data type (float64, float32, float16)
   */
  abstract normalize(
    mean: Tensor<DTpe>,
    variance: Tensor<DTpe>,
    epsilon: number,
    scale: Tensor<DTpe>,
    bias: Tensor<DTpe>
  ): Tensor<DTpe>;

  abstract add_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor<DTpe>;

  abstract subtract_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number,
    beta: number
  ): Tensor<DTpe>;

  abstract multiply_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number
  ): Tensor<DTpe>;

  abstract divide_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[],
    alpha: number
  ): Tensor<DTpe>;

  abstract power_impl(
    th: Tensor<DTpe>,
    tensor: Tensor<DTpe>,
    resultShape: readonly number[]
  ): Tensor<DTpe>;

  abstract gemm_impl(
    b: Tensor<DTpe>,
    aTranspose: boolean,
    bTranspose: boolean,
    alpha: number,
    beta: number,
    C?: Tensor<DTpe>
  ): Tensor<DTpe>;

  protected abstract sum_impl(axes: number[], keepDims: boolean): Tensor<DTpe>;
  protected abstract sumSquare_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor<DTpe>;

  protected abstract product_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor<DTpe>;

  protected abstract max_impl(axes: number[], keepDims: boolean): Tensor<DTpe>;

  protected abstract min_impl(axes: number[], keepDims: boolean): Tensor<DTpe>;

  protected abstract argMax_impl(
    axes: number[],
    selectLast: boolean
  ): Tensor<'uint32'>;

  protected abstract argMin_impl(
    axes: number[],
    selectLast: boolean
  ): Tensor<'uint32'>;

  protected abstract reduceMean_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor<DTpe>;

  protected abstract reduceMeanSquare_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor<DTpe>;

  protected abstract reduceLogSum_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor<DTpe>;

  protected abstract reduceLogSumExp_impl(
    axes: number[],
    keepDims: boolean
  ): Tensor<DTpe>;

  protected abstract conv_impl(
    kernel: Tensor<DTpe>,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[],
    activation: Activation,
    bias?: Tensor<DTpe>
  ): Tensor<DTpe>;

  protected abstract convTranspose_impl(
    kernel: Tensor<DTpe>,
    dilations: number[],
    group: number,
    pads: number[],
    strides: number[]
  ): Tensor<DTpe>;

  protected abstract pad_impl(
    pads: number[],
    mode: PadMode,
    value: number
  ): Tensor<DTpe>;

  protected abstract averagePool_impl(
    kernelShape: number[],
    pads: number[],
    strides: number[],
    includePad: boolean
  ): Tensor<DTpe>;

  protected abstract transpose_impl(permutation: number[]): Tensor<DTpe>;

  protected abstract slice_impl(
    starts: number[],
    ends: number[],
    axes: number[],
    steps: number[]
  ): Tensor<DTpe>;
}
