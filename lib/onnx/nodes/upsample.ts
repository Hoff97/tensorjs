import {Mode} from '../../model/module';
import {CPUTensor} from '../../tensor/cpu/tensor';
import Tensor from '../../types';
import {toCPU} from '../../util/convert';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class UpsampleNode extends OnnxNode {
  private sampleMode: string;
  private scales?: number[];

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    //@ts-ignore
    this.sampleMode = this.getAttributeString('mode');

    if (this.sampleMode !== 'nearest') {
      throw new Error(
        'Upsampling only supported with nearest neighbor sampling'
      );
    }

    if (this.onnxVersion < 9) {
      const scales = this.getAttributeFloats('scales');
      if (scales !== undefined && scales !== null) {
        this.scales = scales;
      } else {
        throw new Error(
          `Upsample node with onnx version ${this.onnxVersion} is missing scales attribute`
        );
      }
    }
  }

  async getScales(scale: Tensor<any>): Promise<number[]> {
    if (this.onnxVersion < 9) {
      return this.scales as number[];
    }

    if (!(scale instanceof CPUTensor)) {
      console.warn('Scales tensor for upsample not on CPU, need to transfer!');
      scale = await toCPU(scale);
    }

    const sc = scale as CPUTensor<any>;

    const scales = new Array(sc.size);
    for (let i = 0; i < sc.size; i++) {
      scales[i] = sc.get(i);
    }
    return scales;
  }

  async forward(inputs: Tensor<any>[]): Promise<Tensor<any>[]> {
    const x = inputs[0];
    const scale = inputs[1];

    const scales = await this.getScales(scale);

    return [x.upsample(scales)];
  }

  getType() {
    return 'Upsample';
  }

  delete(): void {}
}
