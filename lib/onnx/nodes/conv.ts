import Tensor, {Activation, Precision} from '../../types';
import {toCPU, toGPU, toWASM} from '../../util/convert';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class ConvNode extends OnnxNode {
  private group: number;
  private dilations?: number[];
  private pads?: number[];
  private strides?: number[];

  public kernel?: Tensor;
  public bias?: Tensor;

  private activation: Activation;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    kernel?: Tensor,
    bias?: Tensor,
    activation?: Activation
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    const autoPad = this.getAttributeString('autoPad');
    if (autoPad !== undefined) {
      throw new Error('Autopad in conv not supported yet');
    }

    if (activation === undefined) {
      activation = 'id';
    }
    this.activation = activation;

    this.group = this.getAttributeInt('group') || 1;
    this.dilations = this.getAttributeInts('dilations');
    this.pads = this.getAttributeInts('pads');
    this.strides = this.getAttributeInts('strides');

    this.kernel = kernel;
    this.bias = bias;
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];
    const w = this.kernel !== undefined ? this.kernel : inputs[1];
    const b = inputs.length > 2 ? inputs[2] : this.bias;

    return [
      x.conv(
        w,
        b,
        this.dilations,
        this.group,
        this.pads,
        this.strides,
        this.activation
      ),
    ];
  }

  getDilations(rank: number) {
    if (this.dilations !== undefined) {
      return this.dilations;
    }
    return new Array(rank).fill(1);
  }

  getPads(rank: number) {
    if (this.pads !== undefined) {
      return this.pads;
    }
    return new Array(rank * 2).fill(0);
  }

  getStrides(rank: number) {
    if (this.strides !== undefined) {
      return this.strides;
    }
    return new Array(rank).fill(1);
  }

  getType() {
    return 'Conv';
  }

  async toCPU() {
    if (this.kernel !== undefined) {
      this.kernel = await toCPU(this.kernel);
    }
    if (this.bias !== undefined) {
      this.bias = await toCPU(this.bias);
    }
  }

  async toWASM() {
    if (this.kernel !== undefined) {
      this.kernel = await toWASM(this.kernel);
    }
    if (this.bias !== undefined) {
      this.bias = await toWASM(this.bias);
    }
  }

  async toGPU(precision?: Precision) {
    if (precision === undefined) {
      precision = 32;
    }
    if (this.kernel !== undefined) {
      this.kernel = await toGPU(this.kernel, precision);
    }
    if (this.bias !== undefined) {
      this.bias = await toGPU(this.bias, precision);
    }
  }

  delete(): void {
    if (this.kernel !== undefined) {
      this.kernel.delete();
    }
    if (this.bias !== undefined) {
      this.bias.delete();
    }
  }
}
