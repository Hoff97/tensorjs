import {Mode} from '../../model/module';
import {CPUTensor} from '../../tensor/cpu/tensor';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class TileNode extends OnnxNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);
  }

  async forward(inputs: Tensor<any>[]): Promise<Tensor<any>[]> {
    const x = inputs[0];
    const repeats = inputs[1];

    if (!(repeats instanceof CPUTensor)) {
      throw new Error('Tile only works with CPU tensor as repeats');
    }

    if (this.onnxVersion < 13 && this.onnxVersion >= 6) {
      const _repeats = new Array(repeats.size);
      for (let i = 0; i < repeats.size; i++) {
        _repeats[i] = repeats.get(i);
      }

      return [x.repeat(_repeats)];
    }
    throw new Error(
      `Tile with onnx version ${this.onnxVersion} not yet implemented`
    );
  }

  getType() {
    return 'Tile';
  }

  delete(): void {}
}
