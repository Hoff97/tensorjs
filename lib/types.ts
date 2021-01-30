import { CPUTensor } from './tensor/cpu/tensor';
import { compareShapes, getSize } from './util/shape';

export type PadMode = 'constant' | 'reflect' | 'edge';

export type TensorValues = Float32Array | Int32Array;

export default abstract class Tensor {
  abstract getValues(): Promise<TensorValues>;

  abstract getShape(): ReadonlyArray<number>;

  abstract delete(): void;

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

  getAxes(axes?: number | number[]) {
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

  sum(axes?: number | number[], keepDims?: boolean): Tensor {
    let ax = this.getAxes(axes);
    keepDims = keepDims || false;
    return this.sum_impl(ax, keepDims);
  }

  sumSquare(axes?: number | number[], keepDims?: boolean): Tensor {
    let ax = this.getAxes(axes);
    keepDims = keepDims || false;
    return this.sumSquare_impl(ax, keepDims);
  }

  product(axes?: number | number[], keepDims?: boolean): Tensor {
    let ax = this.getAxes(axes);
    keepDims = keepDims || false;
    return this.product_impl(ax, keepDims);
  }

  max(axes?: number | number[], keepDims?: boolean): Tensor {
    let ax = this.getAxes(axes);
    keepDims = keepDims || false;
    return this.max_impl(ax, keepDims);
  }

  min(axes?: number | number[], keepDims?: boolean): Tensor {
    let ax = this.getAxes(axes);
    keepDims = keepDims || false;
    return this.min_impl(ax, keepDims);
  }

  reduceMean(axes?: number | number[], keepDims?: boolean): Tensor {
    let ax = this.getAxes(axes);
    keepDims = keepDims || false;

    return this.reduceMean_impl(ax, keepDims);
  }

  reduceMeanSquare(axes?: number | number[], keepDims?: boolean): Tensor {
    let ax = this.getAxes(axes);
    keepDims = keepDims || false;

    return this.reduceMeanSquare_impl(ax, keepDims);
  }

  conv(kernel: Tensor,
       bias?: Tensor,
       dilations?: number[],
       group?: number,
       pads?: number[],
       strides?: number[],
       activation?: Activation): Tensor {
    const sh = this.getShape();
    const dataRank = sh.length - 2;

    dilations = dilations || new Array(dataRank).fill(1);
    group = group || 1;
    pads = pads || new Array(dataRank * 2).fill(0);
    strides = strides || new Array(dataRank).fill(1);

    if (activation === undefined) {
      activation = "id";
    }

    return this.conv_impl(kernel, dilations, group, pads, strides, bias, activation);
  }

  pad(pads: number[],
      mode?: PadMode,
      value?: number): Tensor {
    if (mode === undefined) {
      mode = 'constant';
    }
    if (value === undefined) {
      value = 0;
    }
    return this.pad_impl(pads, mode, value);
  }

  averagePool(kernelShape: number[],
               pads?: number[],
               strides?: number[],
               includePad?: boolean): Tensor {
    const sh = this.getShape();
    const dataRank = sh.length - 2;

    pads = pads || new Array(dataRank * 2).fill(0);
    strides = strides || new Array(dataRank).fill(1);
    includePad = includePad || false;

    return this.averagePool_impl(kernelShape, pads, strides, includePad);
  }

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

  abstract reshape_impl(shape: readonly number[], copy: boolean): Tensor;

  abstract exp(): Tensor;

  abstract log(): Tensor;

  abstract sqrt(): Tensor;

  abstract abs(): Tensor;

  alignShapes(shape1: readonly number[], shape2: readonly number[]): (readonly number[])[] {
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

  alignTensor(tensor: Tensor) {
    let thisShape = this.getShape();
    let thatShape = tensor.getShape();
    if (compareShapes(thisShape, thatShape)) {
      return [this, tensor, thisShape];
    }
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

  add(tensor: Tensor) {
    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.add_impl(th as Tensor, tens as Tensor, resultShape as number[]);
  }

  subtract(tensor: Tensor) {
    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.subtract_impl(th as Tensor, tens as Tensor, resultShape as number[]);
  }

  multiply(tensor: Tensor) {
    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.multiply_impl(th as Tensor, tens as Tensor, resultShape as number[]);
  }

  divide(tensor: Tensor) {
    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.divide_impl(th as Tensor, tens as Tensor, resultShape as number[]);
  }

  power(tensor: Tensor) {
    const [th, tens, resultShape] = this.alignTensor(tensor);
    return this.power_impl(th as Tensor, tens as Tensor, resultShape as number[]);
  }

  transpose(permutation?: number[]): Tensor {
    if (permutation === undefined) {
      const shape = this.getShape();
      const rank = shape.length;
      permutation = [];
      for (let i = 0; i < rank; i++) {
        permutation.push(rank - i - 1)
      }
    }
    return this.transpose_impl(permutation);
  }

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

  gemm(b: Tensor, aTranspose?: boolean, bTranspose?: boolean,
       alpha?: number, c?: Tensor, beta?: number): Tensor {
    aTranspose = aTranspose || false;
    bTranspose = bTranspose || false;
    alpha = alpha !== undefined ? alpha : 1;
    beta = beta !== undefined ? beta : 1;

    if (c !== undefined) {
      const aShape = this.getShape();
      let cShape = c.getShape();
      const aRank = aShape.length;
      const cRank = cShape.length;

      cShape = [...new Array(aRank - cRank).fill(1), ...cShape];
      c = c.reshape(cShape, false);
    }

    return this.gemm_impl(b, aTranspose, bTranspose, alpha, beta, c);
  }

  slice(starts: number[], ends: number[], axes?: number[]): Tensor {
    const shape = this.getShape();
    const rank = shape.length;
    if (axes === undefined) {
      axes = [];
      for (let i = 0; i < rank; i++) {
        axes.push(i)
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

  abstract add_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor;

  abstract subtract_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor;

  abstract multiply_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor;

  abstract divide_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor;

  abstract power_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor;

  abstract matMul(tensor: Tensor): Tensor;

  abstract gemm_impl(b: Tensor, aTranspose: boolean, bTranspose: boolean,
                     alpha: number, beta: number, C?: Tensor): Tensor;

  abstract sum_impl(axes: number[], keepDims: boolean): Tensor;
  abstract sumSquare_impl(axes: number[], keepDims: boolean): Tensor;

  abstract product_impl(axes: number[], keepDims: boolean): Tensor;

  abstract max_impl(axes: number[], keepDims: boolean): Tensor;

  abstract min_impl(axes: number[], keepDims: boolean): Tensor;

  abstract reduceMean_impl(axes: number[], keepDims: boolean): Tensor;

  abstract reduceMeanSquare_impl(axes: number[], keepDims: boolean): Tensor;

  abstract conv_impl(kernel: Tensor,
                     dilations: number[],
                     group: number,
                     pads: number[],
                     strides: number[],
                     bias?: Tensor,
                     activation?: Activation): Tensor;

  abstract pad_impl(pads: number[],
                    mode: PadMode,
                    value: number): Tensor;

  abstract averagePool_impl(kernelShape: number[],
                             pads: number[],
                             strides: number[],
                             includePad: boolean): Tensor;

  abstract concat(tensor: Tensor, axis: number): Tensor;

  abstract transpose_impl(permutation: number[]): Tensor;

  abstract clip(min?: number, max?: number): Tensor;

  abstract repeat(repeats: number[]): Tensor;

  abstract expand(shape: number[]): Tensor;

  abstract copy(): Tensor;

  abstract gather(axis: number, indices: CPUTensor): Tensor;

  abstract floor(): Tensor;

  abstract ceil(): Tensor;

  abstract slice_impl(starts: number[], ends: number[], axes: number[]): Tensor;

  abstract upsample(scales: number[]): Tensor;

  abstract normalize(mean: Tensor, variance: Tensor, epsilon: number, scale: Tensor, bias: Tensor): Tensor;
}

export type Activation = "id" | "relu" | "relu6";

export type Precision = 16 | 32;