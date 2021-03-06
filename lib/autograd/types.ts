import {Tensor} from '../library';
import {DType} from '../types';

export interface BackwardOp<DTpe extends DType> {
  backward(grad: Tensor<DTpe>): void;

  delete(): void;
}

export interface VariableI<DTpe extends DType> extends Tensor<DTpe> {
  grad?: Tensor<DTpe>;
  value: Tensor<DTpe>;

  noGrad: boolean;

  backward(grad: Tensor<DTpe>): boolean;

  isLeaf(): boolean;
  delete(): void;
}
