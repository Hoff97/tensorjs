import {Module} from '../module';
import {Optimizer} from './optimizer';

/**
 * Stochastic gradient descent optimizer
 */
export class SGD extends Optimizer {
  /**
   * New SGD optimizer for a particular model.
   *
   * @param lr Learning rate, the step size for each update step
   */
  constructor(model: Module, public lr = 0.001) {
    super(model);
  }

  step(): void {
    for (const parameter of this.parameters) {
      if (parameter.grad !== undefined) {
        const oldValue = parameter.value;

        parameter.value = parameter.value.subtract(parameter.grad, 1, this.lr);

        oldValue.delete();
      }
    }
  }
}
