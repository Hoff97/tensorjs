import { RepeatInfo, RepeatOperation } from "../../ops/gpu/repeat";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class TileNode extends OnnxNode {
  private compiled = false;
  private operation?: RepeatOperation<GPUTensor>;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
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

      if (!this.compiled) {
        return [x.repeat(_repeats)];
      } else {
        return [this.operation.calc({
          A: x as GPUTensor,
          repeats: _repeats
        })];
      }
    }
    throw new Error(`Tile with onnx version ${this.onnxVersion} not yet implemented`);
  }

  async staticForward(inputs: Tensor[], compile: boolean): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    if (this.onnxVersion < 13 && this.onnxVersion >= 6) {
      const x = inputs[0];
      const repeats = inputs[1];

      if (!(repeats instanceof CPUTensor)) {
        throw new Error('Tile only works with CPU tensor as repeats');
      }

      const _repeats = new Array(repeats.size);
      for (let i = 0; i < repeats.size; i++) {
        _repeats[i] = repeats.get(i);
      }

      const resultShape = this.operation.getOutputShape({A: x as any, repeats: _repeats});
      const memory = this.allocator.allocate(getSize(resultShape));

      if (compile) {
        const xMem = (x as any).memory;

        const info: RepeatInfo = {
          shapeA: x.getShape(),
          widthA: xMem.width,
          heightA: xMem.height,

          shapeOutput: resultShape,
          widthOutput: memory.width,
          heightOutput: memory.height,

          repeats: _repeats
        };

        this.operation.compile(info);

        this.compiled = true;
      }

      return {
        outputs: [new PrototypeTensor(resultShape, memory)]
      };
    }
    throw new Error(`Tile is not implemented for onnx version ${this.onnxVersion}`);
  }

  initializeForCompiling(): void {
    this.operation = new RepeatOperation(gpuConstructor, this.allocator);
  }

  getType() {
    return 'Tile';
  }
}