import {Variable} from '../../autograd/variable';
import {Module} from '../module';

/**
 * Base class for all gradient based model optimizers.
 */
export abstract class Optimizer {
  parameters: Variable<any>[];

  /**
   * Construct a new optimizer for a particular model
   */
  constructor(public model: Module) {
    this.parameters = model.getParameters();
  }

  /**
   * Make one optimization step. This assumes that gradients where
   * computed previously by calling backward on a scalar variable before.
   */
  abstract step(): void;

  /**
   * Zeros all gradients of the model parameters. This should be called
   * after each optimization step, when the gradients are no longer needed.
   */
  zeroGrads() {
    for (const parameter of this.parameters) {
      if (parameter.grad !== undefined) {
        parameter.grad.delete();
        parameter.grad = undefined;
      }
    }
  }
}
