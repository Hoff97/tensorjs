import {Mode} from '../../../model/module';
import Tensor from '../../../types';
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

  compute(x: Tensor): Tensor {
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

  compute(x: Tensor): Tensor {
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

  compute(x: Tensor): Tensor {
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

  compute(x: Tensor): Tensor {
    return x.asinh();
  }
}
