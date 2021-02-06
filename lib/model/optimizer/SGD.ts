import {Tensor} from '../../library';
import {CPUTensor} from '../../tensor/cpu/tensor';
import {toBackend} from '../../util/convert';
import {Module} from '../module';
import {Optimizer} from './optimizer';

export class SGD extends Optimizer {
  public lrTensor: Tensor;

  constructor(model: Module, public lr = 0.001) {
    super(model);
  }

  step(): void {
    for (const parameter of this.parameters) {
      if (parameter.grad !== undefined) {
        const oldValue = parameter.value;

        parameter.value = parameter.value.subtract(parameter.grad, this.lr);

        oldValue.delete();
      }
    }
  }
}
