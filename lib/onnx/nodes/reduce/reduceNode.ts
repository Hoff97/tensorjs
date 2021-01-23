import { Tensor } from "../../../library";
import { PoolOperation } from "../../../ops/gpu/pool";
import { PrototypeTensor } from "../../../tensor/cpu/prototype";
import { CPUTensor } from "../../../tensor/cpu/tensor";
import { GPUTensor } from "../../../tensor/gpu/tensor";
import types from "../../../types";
import { getSize } from "../../../util/shape";
import { OnnxNode } from "../../node";
import { Attributes, Constants } from "../../types";

export abstract class ReduceNode extends OnnxNode {
  protected axes?: number[];
  protected keepDims?: boolean;

  protected compiled: boolean = false;

  protected operation?: PoolOperation<GPUTensor>;

  protected name: string;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.axes = this.getAttributeInts("axes");
    const keep = this.getAttributeInt("keepdims");

    this.keepDims = keep === 1 || keep === undefined;
  }

  abstract calc(input: Tensor): Tensor;

  protected getAxes(input: Tensor) {
    if (this.axes !== undefined) {
      return this.axes;
    } else {
      const rank = input.getShape().length;

      const res = new Array(rank);
      for (let i = 0; i < rank; i++) {
        res[i] = i;
      }
      return res;
    }
  }

  async forward(inputs: types[]): Promise<types[]> {
    if (this.onnxVersion < 11) {

      if (!this.compiled) {
        return [this.calc(inputs[0])];
      } else {
        const axes = this.getAxes(inputs[0]);

        return [this.operation.calc({X: inputs[0] as GPUTensor, axes, keepDims: this.keepDims})];
      }
    }
    throw new Error(`${this.name} is not implemented for onnx version ${this.onnxVersion}`);
  }

  async staticForward(inputs: types[], compile: boolean): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    if (this.onnxVersion < 11) {
      const a = inputs[0];

      const axes = this.getAxes(a);

      const resultShape = this.operation.getOutputShape({X: a as any, axes, keepDims: this.keepDims});

      const memory = this.allocator.allocate(getSize(resultShape));

      if (compile) {
        const [aMem] = this.getMemoryEntries(inputs);

        const info = {
          shapeA: a.getShape(),
          widthA: aMem.width,
          heightA: aMem.height,
          shapeOutput: resultShape,
          widthOutput: memory.width,
          heightOutput: memory.height,

          axes,
          keepDims: this.keepDims
        };

        this.operation.compile(info);

        this.compiled = true;
      }

      return {
        outputs: [new PrototypeTensor(resultShape, memory)]
      };
    }
    throw new Error(`${this.name} is not implemented for onnx version ${this.onnxVersion}`);
  }

  abstract getOperation(): PoolOperation<GPUTensor>;

  initializeForCompiling(): void {
    this.operation = this.getOperation();
  }
}