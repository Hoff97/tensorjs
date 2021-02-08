import {Variable} from '../../autograd';
import {Mode} from '../../model/module';
import Tensor, {Precision} from '../../types';
import {toCPU, toGPU, toWASM} from '../../util/convert';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';
import {createTensor} from '../util';

export class ConstantNode extends OnnxNode {
  public tensor?: Tensor;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    if (onnxVersion < 11) {
      const tensor = this.getAttributeTensor('value');
      if (tensor !== undefined && tensor !== null) {
        this.tensor = createTensor(tensor);

        if (mode === 'train') {
          this.tensor = new Variable(this.tensor);
        }
      }
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 11 && this.tensor !== undefined) {
      return [this.tensor];
    }
    throw new Error('Constant with onnx version >= 11 not yet implemented');
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

  async toGPU(precision: Precision) {
    if (this.tensor !== undefined) {
      this.tensor = await toGPU(this.tensor, precision);
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
