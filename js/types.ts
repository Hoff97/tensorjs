import { compareShapes } from './util/shape';

export default abstract class Tensor {
  abstract getValues(): Promise<Float32Array>;

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

  sum(axes?: number | number[]): Tensor {
    let ax = this.getAxes(axes);

    return this.sum_impl(ax);
  }

  product(axes?: number | number[]): Tensor {
    let ax = this.getAxes(axes);

    return this.product_impl(ax);
  }

  max(axes?: number | number[]): Tensor {
    let ax = this.getAxes(axes);

    return this.max_impl(ax);
  }

  min(axes?: number | number[]): Tensor {
    let ax = this.getAxes(axes);

    return this.min_impl(ax);
  }

  conv(kernel: Tensor,
       bias?: Tensor,
       dilations?: number[],
       group?: number,
       pads?: number[],
       strides?: number[]): Tensor {
    const sh = this.getShape();
    const dataRank = sh.length - 2;
    
    dilations = dilations || new Array(dataRank).fill(1);
    group = group || 1;
    pads = pads || new Array(dataRank * 2).fill(0);
    strides = strides || new Array(dataRank).fill(1);

    return this.conv_impl(kernel, dilations, group, pads, strides, bias);
  }

  abstract reshape(shape: readonly number[]): Tensor;

  abstract exp(): Tensor;

  abstract log(): Tensor;

  abstract sqrt(): Tensor;

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
      th = this.reshape(thisShape);
    } else if (thatShape.length < thisShape.length) {
      thatShape = [...thatShape];
      const prepend = thisShape.length - thatShape.length;
      (thatShape as number[]).unshift(...new Array(prepend).fill(1));
      tensor = tensor.reshape(thatShape);
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

  abstract add_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor;

  abstract subtract_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor;

  abstract multiply_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor;

  abstract divide_impl(th: Tensor, tensor: Tensor, resultShape: readonly number[]): Tensor;

  abstract matMul(tensor: Tensor): Tensor;

  abstract sum_impl(axes: number[]): Tensor;

  abstract product_impl(axes: number[]): Tensor;

  abstract max_impl(axes: number[]): Tensor;

  abstract min_impl(axes: number[]): Tensor;

  abstract conv_impl(kernel: Tensor,
                     dilations: number[],
                     group: number,
                     pads: number[],
                     strides: number[],
                     bias?: Tensor): Tensor;
}
