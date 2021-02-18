import {Mode} from '../../../model/module';
import Tensor from '../../../types';
import {Attributes, Constants} from '../../types';
import {UnaryNode} from './unaryNode';

export class TanNode extends UnaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Tan', mode);
  }

  compute(x: Tensor): Tensor {
    return x.tan();
  }
}

export class ATanNode extends UnaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Atan', mode);
  }

  compute(x: Tensor): Tensor {
    return x.atan();
  }
}

export class TanHNode extends UnaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'TanH', mode);
  }

  compute(x: Tensor): Tensor {
    return x.tanh();
  }
}

export class ATanHNode extends UnaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'ATanH', mode);
  }

  compute(x: Tensor): Tensor {
    return x.atanh();
  }
}
