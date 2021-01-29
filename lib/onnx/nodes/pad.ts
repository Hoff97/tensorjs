import { PadInfo, PadOperation } from "../../ops/gpu/pad";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor, { PadMode, Precision } from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class PadNode extends OnnxNode {
  private mode: PadMode;
  private pads: number[];
  private value: number;

  private compiled = false;
  private operation?: PadOperation<GPUTensor>;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.mode = (this.getAttributeString('mode') || 'constant') as PadMode;
    this.pads = this.getAttributeInts('pads');
    this.value = this.getAttributeFloat("value") || 0;
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion < 11) {
      if (!this.compiled) {
        return [inputs[0].pad(this.pads, this.mode, this.value)];
      } else {
        return [this.operation.calc({
          input: inputs[0] as GPUTensor,
          mode: this.mode,
          pads: this.pads,
          value: this.value
        })];
      }
    }

    throw new Error(`Pad not implemented for onnx version ${this.onnxVersion}`);
  }

  async staticForward(inputs: Tensor[], compile: boolean, precision: Precision): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    if (this.onnxVersion < 11) {
      const x = inputs[0];

      const resultShape = this.operation.getOutputShape({input: x as GPUTensor, mode: this.mode, pads: this.pads, value: this.value});
      const memory = this.allocator.allocate(getSize(resultShape), precision);

      if (compile) {
        const [xMem] = this.getMemoryEntries(inputs);

        const info: PadInfo = {
          shapeX: x.getShape(),
          widthX: xMem.width,
          heightX: xMem.height,

          shapeOutput: resultShape,
          widthOutput: memory.width,
          heightOutput: memory.height,

          mode: this.mode, pads: this.pads, value: this.value
        };

        this.operation.compile(info, precision);

        this.compiled = true;
      }

      return {
        outputs: [new PrototypeTensor(resultShape, memory)]
      };
    }
    throw new Error(`Pad is not implemented for onnx version ${this.onnxVersion}`);
  }

  initializeForCompiling(): void {
    this.operation = new PadOperation(gpuConstructor, this.allocator);
  }

  getType() {
    return 'Pad';
  }

  delete(): void {}
}