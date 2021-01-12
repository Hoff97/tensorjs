import Tensor from "../../types";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class ConvNode extends OnnxNode {
  private group: number;
  private dilations?: number[];
  private pads?: number[];
  private strides?: number[];

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    const autoPad = this.getAttributeString('autoPad');
    if (autoPad !== undefined) {
      throw new Error('Autopad in conv not supported yet');
    }

    this.group = this.getAttributeInt('group') || 1;
    this.dilations = this.getAttributeInts('dilations');
    this.pads = this.getAttributeInts('pads');
    this.strides = this.getAttributeInts('strides');
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];
    const w = inputs[1];
    const b = inputs.length > 2 ? inputs[2] : undefined;

    return [x.conv(w, b, this.dilations, this.group, this.pads, this.strides)];
  }
}