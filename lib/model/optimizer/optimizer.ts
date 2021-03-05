import {Variable} from '../../autograd/variable';
import {Module} from '../module';

export abstract class Optimizer {
  parameters: Variable<any>[];

  constructor(public model: Module) {
    this.parameters = model.getParameters();
  }

  abstract step(): void;

  zeroGrads() {
    for (const parameter of this.parameters) {
      if (parameter.grad !== undefined) {
        parameter.grad.delete();
        parameter.grad = undefined;
      }
    }
  }
}
