import Tensor, { PadMode } from "../../types";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class PadNode extends OnnxNode {
  private mode: PadMode;
  private pads: number[];
  private value: number;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.mode = (this.getAttributeString('mode') || 'constant') as PadMode;
    this.pads = this.getAttributeInts('pads');
    this.value = this.getAttributeFloat("value") || 0;
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 11) {
      return [inputs[0].pad(this.pads, this.mode, this.value)];
    }

    throw new Error(`Pad not implemented for onnx version ${this.onnxVersion}`);
  }

  getType() {
    return 'Pad';
  }

  delete(): void {}
}