import {Mode} from '../../../model/module';
import Tensor, {DType} from '../../../types';
import {Attributes, Constants} from '../../types';
import {UnaryNode} from './unaryNode';

export class SinNode extends UnaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Sin', mode);
  }

  compute<DTpe extends DType>(x: Tensor<DTpe>): Tensor<DTpe> {
    return x.sin();
  }
}

export class ASinNode extends UnaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Asin', mode);
  }

  compute<DTpe extends DType>(x: Tensor<DTpe>): Tensor<DTpe> {
    return x.asin();
  }
}

export class SinHNode extends UnaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Sinh', mode);
  }

  compute<DTpe extends DType>(x: Tensor<DTpe>): Tensor<DTpe> {
    return x.sinh();
  }
}

export class ASinHNode extends UnaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Asinh', mode);
  }

  compute<DTpe extends DType>(x: Tensor<DTpe>): Tensor<DTpe> {
    return x.asinh();
  }
}
