import {Tensor} from '../library';

export interface BackwardOp {
  backward(grad: Tensor): void;
}

export interface VariableI extends Tensor {
  grad?: Tensor;
  value: Tensor;

  backward(grad: Tensor): void;
}
