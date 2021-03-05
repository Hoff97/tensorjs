import {DTypeGpu} from '../../../tensor/gpu/interface';
import {GPUTensor} from '../../../tensor/gpu/tensor';
import Tensor, {DType} from '../../../types';
import {Module} from '../../module';
import {Optimizer} from '../optimizer';
import {defaultUpdateMomentsD} from './updateMoments';
import {defaultUpdateValueD} from './updateParams';

/**
 * Implements the Adam optimizer
 *
 * This is currently quite slow on the CPU and WASM backends. On the GPU
 * backend, one update step is only slightly slower than an update step of SGD
 * and will converge a lot quicker.
 */
export class Adam extends Optimizer {
  public moment1?: (Tensor<any> | undefined)[];
  public moment2?: (Tensor<any> | undefined)[];

  public moments?: GPUTensor<any>[];

  public t = 0;

  constructor(
    model: Module,
    public lr = 0.001,
    public beta1 = 0.9,
    public beta2 = 0.999,
    public epsilon = 10e-8
  ) {
    super(model);

    const params = this.parameters;
    if (params[0].value instanceof GPUTensor) {
      this.moments = new Array(params.length);
      for (let i = 0; i < params.length; i++) {
        this.moments[i] = new GPUTensor(
          new Array((params[i].value as GPUTensor<any>).size * 4).fill(0),
          [...params[i].getShape(), 4],
          params[i].value.dtype
        );
      }
    } else {
      this.moment1 = new Array(params.length).fill(undefined);
      this.moment2 = new Array(params.length).fill(undefined);
    }
  }

  step(): void {
    this.t++;

    if (this.moment1 !== undefined && this.moment2 !== undefined) {
      for (let i = 0; i < this.parameters.length; i++) {
        const parameter = this.parameters[i];
        if (parameter.grad !== undefined) {
          const oldValue = parameter.value;

          const {newValue, moment1, moment2} = this.paramStep(
            parameter.value,
            parameter.grad,
            this.moment1[i],
            this.moment2[i]
          );
          parameter.value = newValue;
          this.moment1[i] = moment1;
          this.moment2[i] = moment2;

          oldValue.delete();
        }
      }
    } else if (this.moments !== undefined) {
      for (let i = 0; i < this.parameters.length; i++) {
        const parameter = this.parameters[i];
        if (parameter.grad !== undefined) {
          const oldValue = parameter.value;

          const {newValue, moments} = this.gpuParamStep(
            parameter.value as GPUTensor<any>,
            parameter.grad as GPUTensor<any>,
            this.moments[i]
          );
          parameter.value = newValue as GPUTensor<any>;
          this.moments[i] = moments as GPUTensor<any>;

          oldValue.delete();
        }
      }
    }
  }

  updateMoments<DTpe extends DType>(
    grad: Tensor<DTpe>,
    moment1: Tensor<DTpe> | undefined,
    moment2: Tensor<DTpe> | undefined
  ) {
    let moment1New;
    if (moment1 === undefined) {
      moment1New = grad.multiplyScalar(1 - this.beta1);
    } else {
      const oldMoment1 = moment1;
      moment1New = moment1.add(grad, this.beta1, 1 - this.beta1);
      oldMoment1.delete();
    }
    let moment2New;
    if (moment2 === undefined) {
      moment2New = grad.multiply(grad, 1 - this.beta2);
    } else {
      const gradSquared = grad.multiply(grad);
      const oldMoment2 = moment2;
      moment2New = moment2.add(gradSquared, this.beta2, 1 - this.beta2);
      gradSquared.delete();
      oldMoment2.delete();
    }
    return {moment1New, moment2New};
  }

  getCorrectedMoments<DTpe extends DType>(
    moment1: Tensor<DTpe>,
    moment2: Tensor<DTpe>
  ) {
    const correctMoment1 = moment1.addMultiplyScalar(
      1 / (1 - Math.pow(this.beta1, this.t)),
      0
    );

    const correctMoment2 = moment2.addMultiplyScalar(
      1 / (1 - Math.pow(this.beta2, this.t)),
      0
    );

    return {correctMoment1, correctMoment2};
  }

  paramStep<DTpe extends DType>(
    value: Tensor<DTpe>,
    grad: Tensor<DTpe>,
    moment1: Tensor<DTpe> | undefined,
    moment2: Tensor<DTpe> | undefined
  ) {
    const {moment1New, moment2New} = this.updateMoments(grad, moment1, moment2);

    // This is not 100% correct, in the original paper
    // the epsilon occurs outside of the square root
    // It does not make much of a difference though
    // and is slightly faster
    const correctMoment2 = moment2New.addMultiplyScalar(
      1 / (1 - Math.pow(this.beta2, this.t)),
      this.epsilon
    );
    const moment2Sqrt = correctMoment2.sqrt();
    correctMoment2.delete();

    const step = moment1New.divide(
      moment2Sqrt,
      -this.lr / (1 - Math.pow(this.beta1, this.t))
    );
    moment2Sqrt.delete();

    const newValue = value.add(step);
    step.delete();
    return {newValue, moment1, moment2};
  }

  gpuParamStep<DTpe extends DTypeGpu>(
    value: GPUTensor<DTpe>,
    grad: GPUTensor<DTpe>,
    moments: GPUTensor<DTpe>
  ) {
    const newMoments = defaultUpdateMomentsD.calc(
      {
        Grad: grad,
        Moments: moments,
        beta1: this.beta1,
        beta2: this.beta2,
        t: this.t,
      },
      value.dtype
    );
    moments.delete();

    const newValue = defaultUpdateValueD.calc(
      {
        Value: value,
        Moments: newMoments,
        alpha: this.lr,
        epsilon: this.epsilon,
      },
      value.dtype
    );

    return {newValue, moments: newMoments};
  }
}
