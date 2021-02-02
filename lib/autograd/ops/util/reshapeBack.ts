import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ReshapeBack implements BackwardOp {
  constructor(public a: VariableI) {}

  backward(grad: Tensor): void {
    const shapeA = this.a.getShape();

    //TODO: Should copy=False really be set here?
    this.a.backward(grad.reshape(shapeA, false));
  }
}
