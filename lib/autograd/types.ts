import {Tensor} from '../library';

export interface BackwardOp {
  backward(grad: Tensor): void;

  delete(): void;
}

export interface VariableI extends Tensor {
  grad?: Tensor;
  value: Tensor;

  noGrad: boolean;

  backward(grad: Tensor): void;

  isLeaf(): boolean;
  delete(): void;
}
