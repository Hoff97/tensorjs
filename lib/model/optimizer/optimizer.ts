import {Variable} from '../../autograd/variable';
import {Tensor} from '../../library';
import {Module} from '../module';

export abstract class Optimizer {
  parameters: Variable[];

  constructor(public model: Module) {
    this.parameters = model.getParameters();
  }

  abstract step(): void;

  zeroGrads() {
    for (const parameter of this.parameters) {
      parameter.grad = undefined;
    }
  }
}
