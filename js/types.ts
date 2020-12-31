import { compareShapes } from './util/shape';

export default abstract class Tensor {
  abstract getValues(): Promise<Float32Array>;

  abstract getShape(): ReadonlyArray<number>;

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

  abstract exp(): Tensor;

  abstract log(): Tensor;

  abstract sqrt(): Tensor;

  abstract add(tensor: Tensor): Tensor;

  abstract subtract(tensor: Tensor): Tensor;

  abstract multiply(tensor: Tensor): Tensor;

  abstract divide(tensor: Tensor): Tensor;

  abstract matMul(tensor: Tensor): Tensor;
}
