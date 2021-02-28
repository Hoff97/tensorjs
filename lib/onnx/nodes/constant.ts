import {Variable} from '../../autograd';
import {Mode} from '../../model/module';
import Tensor from '../../types';
import {toCPU, toGPU, toWASM} from '../../util/convert';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';
import {createTensor} from '../util';

export class ConstantNode extends OnnxNode {
  public tensor?: Tensor<any>;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    const tensor = this.getAttributeTensor('value');
    if (tensor !== undefined && tensor !== null) {
      this.tensor = createTensor(tensor);

      if (mode === 'train' && this.tensor !== undefined) {
        this.tensor = new Variable(this.tensor);
      }
    } else {
      throw new Error(
        'Constant needs tensor value, but attribute "value" was not defined'
      );
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async forward(inputs: Tensor<any>[]): Promise<Tensor<any>[]> {
    if (this.tensor !== undefined) {
      return [this.tensor];
    }
    throw new Error('Constant without tensor value doesnt work');
  }

  async toCPU() {
    if (this.tensor !== undefined) {
      this.tensor = await toCPU(this.tensor);
    }
  }

  async toWASM() {
    if (this.tensor !== undefined) {
      this.tensor = await toWASM(this.tensor);
    }
  }

  async toGPU() {
    if (this.tensor !== undefined) {
      this.tensor = await toGPU(this.tensor);
    }
  }

  getType() {
    return 'Constant';
  }

  delete(): void {
    if (this.tensor !== undefined) {
      this.tensor.delete();
    }
  }
}
