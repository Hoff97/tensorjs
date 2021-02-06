import {Tensor} from '../../library';
import {CPUTensor} from '../../tensor/cpu/tensor';
import {toBackend} from '../../util/convert';
import {Module} from '../module';
import {Optimizer} from './optimizer';

export class SGD extends Optimizer {
  public lrTensor: Tensor;

  constructor(model: Module, public lr = 0.001) {
    super(model);

    this.lrTensor = new CPUTensor([1], [lr]);
    toBackend(this.lrTensor, this.model.backend, 32).then(x => {
      this.lrTensor = x;
    });
  }

  step(): void {
    for (const parameter of this.parameters) {
      if (parameter.grad !== undefined) {
        const oldValue = parameter.value;

        const scaled = parameter.grad.multiply(this.lrTensor);
        parameter.value = parameter.value.subtract(scaled);

        scaled.delete();
        oldValue.delete();
      }
    }
  }
}
