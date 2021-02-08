import {Tensor} from '../../library';
import {Module} from '../module';
import {Optimizer} from './optimizer';

/**
 * Implements the Adam optimizer
 *
 * This is currently quite slow, since a lot of computation steps
 * have to happen in the parameter update step
 */
export class Adam extends Optimizer {
  public moment1: (Tensor | undefined)[];
  public moment2: (Tensor | undefined)[];

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
    this.moment1 = new Array(params.length).fill(undefined);
    this.moment2 = new Array(params.length).fill(undefined);
  }

  step(): void {
    this.t++;
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
  }

  paramStep(
    value: Tensor,
    grad: Tensor,
    moment1: Tensor | undefined,
    moment2: Tensor | undefined
  ) {
    if (moment1 === undefined) {
      moment1 = grad.multiplyScalar(1 - this.beta1);
    } else {
      const oldMoment1 = moment1;
      moment1 = moment1.add(grad, this.beta1, 1 - this.beta1);
      oldMoment1.delete();
    }
    if (moment2 === undefined) {
      moment2 = grad.multiply(grad, 1 - this.beta2);
    } else {
      const gradSquared = grad.multiply(grad);
      const oldMoment2 = moment2;
      moment2 = moment2.add(gradSquared, this.beta2, 1 - this.beta2);
      gradSquared.delete();
      oldMoment2.delete();
    }

    // This is not 100% correct, in the original paper
    // the epsilon occurs outside of the square root
    // It does not make much of a difference though
    // and is slightly faster
    const correctMoment2 = moment2.addMultiplyScalar(
      1 / (1 - Math.pow(this.beta2, this.t)),
      this.epsilon
    );
    const moment2Sqrt = correctMoment2.sqrt();
    correctMoment2.delete();

    const step = moment1.divide(
      moment2Sqrt,
      -this.lr / (1 - Math.pow(this.beta1, this.t))
    );
    moment2Sqrt.delete();

    const newValue = value.add(step);
    step.delete();
    return {newValue, moment1, moment2};
  }
}
