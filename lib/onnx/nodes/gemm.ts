import { GemmCOperation, GemmInfo, GemmInput, GemmOperation } from "../../ops/gpu/matMul/gemm";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor, { Precision } from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class GemmNode extends OnnxNode {
  private alpha: number;
  private beta: number;
  private transA: boolean;
  private transB: boolean;

  private compiled = false;
  private operation?: GemmOperation<GPUTensor>;
  private cShape?: readonly number[];

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.alpha = this.getAttributeFloat("alpha") || 1.0;
    this.beta = this.getAttributeFloat("beta") || 1.0;

    const transA = this.getAttributeInt("transA");
    const transB = this.getAttributeInt("transB");

    this.transA = transA === 1;
    this.transB = transB === 1;
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (this.onnxVersion >= 9 && this.onnxVersion < 11) {
      const a = inputs[0];
      const b = inputs[1];
      const c = inputs[2];

      if (!this.compiled) {
        return [a.gemm(b, this.transA, this.transB, this.alpha, c, this.beta)];
      } else {
        const input: GemmInput = {
          a: a as GPUTensor, b: b as GPUTensor,
          aTranspose: this.transA, bTranspose: this.transB,
          alpha: this.alpha, beta: this.beta
        }

        if (c !== undefined) {
          //@ts-ignore
          input['c'] = c.reshape(this.cShape, false);
        }

        return [this.operation.calc(input)];
      }
    }
    throw new Error(`Gemm is not implemented for onnx version ${this.onnxVersion}`);
  }

  async staticForward(inputs: Tensor[], compile: boolean, precision: Precision): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    if (this.onnxVersion < 13 && this.onnxVersion >= 9) {
      if (this.operation === undefined) {
        this.initializeForCompiling(inputs);
      }

      const a = inputs[0];
      const b = inputs[1];
      const c = inputs[2];

      const resultShape = this.operation.getOutputShape({
        a: a as any, b: b as any,
        aTranspose: this.transA, bTranspose: this.transB,
        alpha: this.alpha, beta: this.beta
      });
      const memory = this.allocator.allocate(getSize(resultShape), precision);

      if (compile) {
        const memories = this.getMemoryEntries(inputs);
        const aMem = memories[0];
        const bMem = memories[1];
        const cMem = memories[2];

        let info: GemmInfo = {
          shapeA: a.getShape(),
          widthA: aMem.width,
          heightA: aMem.height,

          shapeB: b.getShape(),
          widthB: bMem.width,
          heightB: bMem.height,

          shapeOutput: resultShape,
          widthOutput: memory.width,
          heightOutput: memory.height,

          aTranspose: this.transA,
          bTranspose: this.transB,
          alpha: this.alpha,
          beta: this.beta
        };

        if (c !== undefined) {
          const aShape = a.getShape();
          let cShape = c.getShape();
          const aRank = aShape.length;
          const cRank = cShape.length;

          this.cShape = [...new Array(aRank - cRank).fill(1), ...cShape];

          info = {
            ...info,
            //@ts-ignore
            shapeC: cShape,
            widthC: cMem.width,
            heightC: cMem.height
          }
        }

        this.operation.compile(info, precision);

        this.compiled = true;
      }

      return {
        outputs: [new PrototypeTensor(resultShape, memory)]
      };
    }
    throw new Error(`Gemm is not implemented for onnx version ${this.onnxVersion}`);
  }

  initializeForCompiling(inputs?: Tensor[]): void {
    if (inputs !== undefined) {
      if (inputs.length === 2) {
        this.operation = new GemmOperation(gpuConstructor, this.allocator);
      } else if (inputs.length === 3) {
        this.operation = new GemmCOperation(gpuConstructor, this.allocator);
      }
    }
  }

  getType() {
    return 'Gemm';
  }

  delete(): void {}
}