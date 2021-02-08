import {Mode} from '../../model/module';
import Tensor, {PadMode} from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class PadNode extends OnnxNode {
  private padMode: PadMode;
  private pads: number[];
  private value: number;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    this.padMode = (this.getAttributeString('mode') || 'constant') as PadMode;
    //@ts-ignore
    this.pads = this.getAttributeInts('pads');
    this.value = this.getAttributeFloat('value') || 0;
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 11) {
      return [inputs[0].pad(this.pads, this.padMode, this.value)];
    }

    throw new Error(`Pad not implemented for onnx version ${this.onnxVersion}`);
  }

  getType() {
    return 'Pad';
  }

  delete(): void {}
}
