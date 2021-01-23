import { ClipInfo, ClipOperation } from "../../ops/gpu/clip";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class ClipNode extends OnnxNode {
  private min?: number;
  private max?: number;

  private compiled = false;
  private operation?: ClipOperation<GPUTensor>;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    if (onnxVersion < 11) {
      this.min = this.getAttributeFloat('min');
      this.max = this.getAttributeFloat('max');
    }
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    if (this.onnxVersion < 11) {
      if (!this.compiled) {
        return [x.clip(this.min, this.max)];
      } else {
        return [this.operation.calc({input: x as GPUTensor, minVal: this.min, maxVal: this.max})];
      }
    } else {
      const min = inputs.length > 1 ? inputs[1] : undefined;
      const max = inputs.length > 2 ? inputs[2] : undefined;
      if (min === undefined && max === undefined) {
        return [x.copy()];
      }
      throw new Error('Clip with onnx version >= 11 not yet implemented');
    }
  }

  async staticForward(inputs: Tensor[], compile: boolean): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    if (this.onnxVersion < 11) {
      const a = inputs[0];
      const resultShape = a.getShape();

      const memory = this.allocator.allocate(getSize(resultShape));

      if (compile) {
        const [aMem] = this.getMemoryEntries(inputs);

        const info: ClipInfo = {
          shapeX: a.getShape(),
          widthX: aMem.width,
          heightX: aMem.height,
          shapeOutput: resultShape,
          widthOutput: memory.width,
          heightOutput: memory.height,

          minVal: this.min !== undefined ? this.min : 0,
          doMin: this.min !== undefined ? 1 : 0,
          maxVal: this.max !== undefined ? this.max : 0,
          doMax: this.max !== undefined ? 1 : 0,
        };

        this.operation.compile(info);

        this.compiled = true;
      }

      return {
        outputs: [new PrototypeTensor(resultShape, memory)]
      };
    }
    throw new Error(`Clip is not implemented for onnx version ${this.onnxVersion}`);
  }

  initializeForCompiling(): void {
    this.operation = new ClipOperation(gpuConstructor, this.allocator);
  }
}