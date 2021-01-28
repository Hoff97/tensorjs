import { ConcatInfo, ConcatOperation } from "../../ops/gpu/concat";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor, { Precision } from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class ConcatNode extends OnnxNode {
  private axis: number;

  private compiled = false;
  private operations?: ConcatOperation<GPUTensor>[];

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    if (onnxVersion < 13) {
      this.axis = this.getAttributeInt("axis");
    }
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (inputs.length > 2) {
      // This logging seems to slow down the operation more than the operation itself
      //console.warn(`Concat with more than 2 tensors is currently slow. Doing concat with ${inputs.length} tensors`);
    }

    if (!this.compiled) {
      let result = inputs[0];
      for (let i = 1; i < inputs.length; i++) {
        let newRes = result.concat(inputs[i], this.axis);
        if (i > 1) {
          result.delete();
        }
        result = newRes;
      }

      return [result];
    } else {
      let result = inputs[0];
      for (let i = 1; i < inputs.length; i++) {
        let newRes = this.operations[i-1].calc({
          A: result as any, B: inputs[i] as any, axis: this.axis
        })
        if (i > 1) {
          (result as GPUTensor).delete(this.allocator);
        }
        result = newRes;
      }

      return [result];
    }
  }

  async staticForward(inputs: Tensor[], compile: boolean, precision: Precision): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    if (this.operations === undefined) {
      this.initializeForCompiling(inputs);
    }

    if (this.onnxVersion < 13) {
      let resShape = inputs[0].getShape();
      const memories = this.getMemoryEntries(inputs);

      let memory = memories[0];

      for (let i = 1; i < inputs.length; i++) {
        const b = inputs[i];

        //@ts-ignore
        const newResShape = this.operations[i-1].getOutputShape({A: {shape: resShape}, B: b, axis: this.axis});
        const newMemory = this.allocator.allocate(getSize(newResShape), precision);

        if (compile) {
          const info: ConcatInfo = {
            shapeA: resShape,
            widthA: memory.width,
            heightA: memory.height,

            shapeB: b.getShape(),
            widthB: memories[i].width,
            heightB: memories[i].height,

            shapeOutput: newResShape,
            widthOutput: newMemory.width,
            heightOutput: newMemory.height,

            axis: this.axis
          };

          this.operations[i-1].compile(info, precision);

          this.compiled = true;
        }

        if (i > 1) {
          this.allocator.deallocate(memory);
        }

        memory = newMemory;
        resShape = newResShape;
      }

      return {
        outputs: [new PrototypeTensor(resShape, memory)]
      };
    }
    throw new Error(`Concat is not implemented for onnx version ${this.onnxVersion}`);
  }

  initializeForCompiling(inputs?: Tensor[]): void {
    if (inputs !== undefined) {
      this.operations = [];
      for (let i = 0; i < inputs.length - 1; i++) {
        this.operations.push(new ConcatOperation(gpuConstructor, this.allocator));
      }
    }
  }

  getType() {
    return 'Concat';
  }
}