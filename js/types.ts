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

  abstract exp(): Tensor;

  abstract log(): Tensor;

  abstract sqrt(): Tensor;

  abstract add(tensor: Tensor): Tensor;

  abstract subtract(tensor: Tensor): Tensor;

  abstract multiply(tensor: Tensor): Tensor;

  abstract divide(tensor: Tensor): Tensor;

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
