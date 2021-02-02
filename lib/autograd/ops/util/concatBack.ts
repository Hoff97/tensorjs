import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ConcatBack implements BackwardOp {
  constructor(public a: VariableI, public b: VariableI, public axis: number) {}

  backward(grad: Tensor): void {
    this.a.backward(
      grad.slice([0], [this.a.getShape()[this.axis]], [this.axis])
    );

    this.b.backward(
      grad.slice(
        [this.a.getShape()[this.axis]],
        [grad.getShape()[this.axis]],
        [this.axis]
      )
    );
  }
}
