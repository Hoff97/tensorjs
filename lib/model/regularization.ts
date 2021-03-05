import {Variable} from '../autograd';
import {Module} from './module';

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
