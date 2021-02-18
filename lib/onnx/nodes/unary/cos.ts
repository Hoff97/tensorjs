import {Mode} from '../../../model/module';
import Tensor from '../../../types';
import {Attributes, Constants} from '../../types';
import {UnaryNode} from './unaryNode';

export class CosNode extends UnaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Cos', mode);
  }

  compute(x: Tensor): Tensor {
    return x.cos();
  }
}

export class ACosNode extends UnaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Acos', mode);
  }

  compute(x: Tensor): Tensor {
    return x.acos();
  }
}

export class CosHNode extends UnaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Cosh', mode);
  }

  compute(x: Tensor): Tensor {
    return x.cosh();
  }
}

export class ACosHNode extends UnaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'AcosH', mode);
  }

  compute(x: Tensor): Tensor {
    return x.acosh();
  }
}
