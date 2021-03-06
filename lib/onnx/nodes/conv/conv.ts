import {Variable} from '../../../autograd';
import {Mode} from '../../../model/module';
import Tensor, {Activation, DType} from '../../../types';
import {toCPU, toGPU, toWASM} from '../../../util/convert';
import {OnnxNode} from '../../node';
import {Attributes, Constants} from '../../types';

export class ConvNode extends OnnxNode {
  private group: number;
  private dilations?: number[];
  private pads?: number[];
  private strides?: number[];

  public kernel?: Tensor<any>;
  public bias?: Tensor<any>;

  private activation: Activation;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode,
    kernel?: Tensor<any>,
    bias?: Tensor<any>,
    activation?: Activation
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

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
    if (mode === 'train' && this.kernel !== undefined) {
      this.kernel = new Variable(this.kernel);
    }
    if (mode === 'train' && this.bias !== undefined) {
      this.bias = new Variable(this.bias);
    }
  }

  async forward<DTpe extends DType>(
    inputs: Tensor<DTpe>[]
  ): Promise<Tensor<DTpe>[]> {
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

  async toGPU() {
    if (this.kernel !== undefined) {
      this.kernel = await toGPU(this.kernel);
    }
    if (this.bias !== undefined) {
      this.bias = await toGPU(this.bias);
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
