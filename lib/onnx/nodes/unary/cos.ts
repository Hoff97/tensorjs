import {Mode} from '../../../model/module';
import Tensor, {DType} from '../../../types';
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

  compute<DTpe extends DType>(x: Tensor<DTpe>): Tensor<DTpe> {
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

  compute<DTpe extends DType>(x: Tensor<DTpe>): Tensor<DTpe> {
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

  compute<DTpe extends DType>(x: Tensor<DTpe>): Tensor<DTpe> {
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

  compute<DTpe extends DType>(x: Tensor<DTpe>): Tensor<DTpe> {
    return x.acosh();
  }
}
