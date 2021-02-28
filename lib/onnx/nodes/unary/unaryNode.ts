import {Mode} from '../../../model/module';
import Tensor, {DType} from '../../../types';
import {OnnxNode} from '../../node';
import {Attributes, Constants} from '../../types';

export abstract class UnaryNode extends OnnxNode {
  protected name: string;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    name: string,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);
    this.name = name;
  }

  abstract compute<DTpe extends DType>(x: Tensor<DTpe>): Tensor<DTpe>;

  async forward<DTpe extends DType>(
    inputs: Tensor<DTpe>[]
  ): Promise<Tensor<DTpe>[]> {
    const x = inputs[0];

    return [this.compute(x)];
  }

  getType() {
    return this.name;
  }

  delete(): void {}
}
