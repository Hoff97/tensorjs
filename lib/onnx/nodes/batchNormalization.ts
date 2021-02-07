import {Variable} from '../../autograd/variable';
import {Mode} from '../../model/module';
import {CPUTensor} from '../../tensor/cpu/tensor';
import Tensor, {Precision} from '../../types';
import {toCPU, toGPU, toWASM} from '../../util/convert';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class BatchNormalizationNode extends OnnxNode {
  private epsilon: number;
  private momentum: number;

  public epsTensor: Tensor;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    this.epsilon = this.getAttributeFloat('epsilon') || 1e-5;
    this.momentum = this.getAttributeFloat('momentum') || 0.9;

    this.epsTensor = new CPUTensor([1], [this.epsilon]);
    if (mode === 'train') {
      this.epsTensor = new Variable(this.epsTensor);
    }

    //TODO: Handle lower onnxversions here
  }

  async defaultForward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    let scale = inputs[1];
    let B = inputs[2];
    let mean = inputs[3];
    let variance = inputs[4];

    //TODO: Handle lower onnx versions here

    const C = scale.getShape()[0];

    const newShape = [1, C, ...new Array(x.getShape().length - 2).fill(1)];

    scale = scale.reshape(newShape, false);
    B = B.reshape(newShape, false);
    mean = mean.reshape(newShape, false);
    variance = variance.reshape(newShape, false);

    const result = x.normalize(mean, variance, this.epsilon, scale, B);

    return [result];
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    return this.defaultForward(inputs);
  }

  getType() {
    return 'BatchNormalization';
  }

  async toCPU() {
    this.epsTensor = await toCPU(this.epsTensor);
  }

  async toWASM() {
    this.epsTensor = await toWASM(this.epsTensor);
  }

  async toGPU(precision: Precision) {
    this.epsTensor = await toGPU(this.epsTensor, precision);
  }

  delete(): void {
    this.epsTensor.delete();
  }
}
