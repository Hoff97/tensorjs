import { ExpOperation } from "../../ops/gpu/exp";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class ExpNode extends OnnxNode {
  protected operation?: ExpOperation<GPUTensor>;

  protected compiled: boolean = false;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    if (!this.compiled) {
      return [x.exp()];
    } else {
      return [this.operation.calc({input: x as GPUTensor})];
    }
  }

  async staticForward(inputs: Tensor[], compile: boolean): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs as CPUTensor[]);
    }

    const x = inputs[0];

    const memory = this.allocator.allocate(getSize(x.getShape()));

    if (compile) {
      const [xMem] = this.getMemoryEntries(inputs);

      const info = {
        shapeX: x.getShape(),
        widthX: xMem.width,
        heightX: xMem.height,
        shapeOutput: x.getShape(),
        widthOutput: memory.width,
        heightOutput: memory.height
      };

      this.operation.compile(info);

      this.compiled = true;
    }

    return {
      outputs: [new PrototypeTensor(x.getShape(), memory)]
    };
  }

  initializeForCompiling(): void {
    this.operation = new ExpOperation(gpuConstructor, this.allocator);
  }
}