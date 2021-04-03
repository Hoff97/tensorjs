import {Variable} from '../autograd';
import {Module} from './module';

/**
 * L2 weight regularization for a particular model.
 *
 * @example
 * ```typescript
 * const model = new Linear(32,1);
 * const regularizer = new L2Regularization(model, 0.01);
 * //...
 * const prediction = (await model.forward([x]))[0];
 * let loss = prediction.subtract(y).reduceSumSquare();
 * loss = loss.add(regularizer.getLoss());
 * //...
 * loss.backward();
 * //...
 * ```
 */
export class L2Regularization {
  public parameters: Variable<any>[];

  constructor(public model: Module, public gamma: number) {
    this.parameters = model.getParameters();
  }

  getLoss() {
    let loss = this.parameters[0].sumSquare();
    let factor = this.gamma;
    for (let i = 1; i < this.parameters.length; i++) {
      loss = loss.add(this.parameters[i].sumSquare(), factor, this.gamma);
      factor = factor / factor;
    }
    return loss;
  }
}

/**
 * L1 weight regularization for a particular model.
 *
 * @example
 * ```typescript
 * const model = new Linear(32,1);
 * const regularizer = new L1Regularization(model, 0.01);
 * //...
 * const prediction = (await model.forward([x]))[0];
 * let loss = prediction.subtract(y).reduceSumSquare();
 * loss = loss.add(regularizer.getLoss());
 * //...
 * loss.backward();
 * //...
 * ```
 */
export class L1Regularization {
  public parameters: Variable<any>[];

  constructor(public model: Module, public gamma: number) {
    this.parameters = model.getParameters();
  }

  getLoss() {
    let loss = this.parameters[0].abs().sum();
    let factor = this.gamma;
    for (let i = 1; i < this.parameters.length; i++) {
      loss = loss.add(this.parameters[i].abs().sum(), factor, this.gamma);
      factor = factor / factor;
    }
    return loss;
  }
}
