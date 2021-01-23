import { ConvBiasOperation, ConvInfo, ConvInput, ConvOperation } from "../../ops/gpu/conv";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor from "../../types";
import { getSize } from "../../util/shape";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class ConvNode extends OnnxNode {
  private group: number;
  private dilations?: number[];
  private pads?: number[];
  private strides?: number[];

  private compiled = false;

  private operation?: ConvOperation<GPUTensor>;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    const autoPad = this.getAttributeString('autoPad');
    if (autoPad !== undefined) {
      throw new Error('Autopad in conv not supported yet');
    }

    this.group = this.getAttributeInt('group') || 1;
    this.dilations = this.getAttributeInts('dilations');
    this.pads = this.getAttributeInts('pads');
    this.strides = this.getAttributeInts('strides');
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (!this.compiled) {
      const x = inputs[0];
      const w = inputs[1];
      const b = inputs.length > 2 ? inputs[2] : undefined;

      return [x.conv(w, b, this.dilations, this.group, this.pads, this.strides)];
    } else {
      const rank = inputs[0].getShape().length - 2;
      const dilations = this.getDilations(rank);
      const pads = this.getPads(rank);
      const strides = this.getStrides(rank);

      let input: ConvInput = {
        X: inputs[0] as GPUTensor,
        W: inputs[1] as GPUTensor,
        pads, dilations, strides
      };

      if (inputs.length > 2) {
        //@ts-ignore
        input['B'] = inputs[2];
      }

      return [this.operation.calc(input)];
    }
  }

  getDilations(rank: number) {
    if (this.dilations !== undefined) {
      return this.dilations;
    }
    return new Array(rank).fill(1);
  }

  getPads(rank: number) {
    if (this.pads !== undefined) {
      return this.pads;
    }
    return new Array(rank*2).fill(0);
  }

  getStrides(rank: number) {
    if (this.strides !== undefined) {
      return this.strides;
    } return new Array(rank).fill(1);
  }

  async staticForward(inputs: Tensor[], compile: boolean): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    if (this.operation === undefined) {
      this.initializeForCompiling(inputs);
    }

    const x = inputs[0];
    const w = inputs[1];

    const bias = inputs.length > 2 ? inputs[2] : undefined;

    const rank = x.getShape().length - 2;
    const dilations = this.getDilations(rank);
    const pads = this.getPads(rank);
    const strides = this.getStrides(rank);

    const resultShape = this.operation.getOutputShape({
      X: inputs[0] as any,
      W: inputs[1] as any,
      pads, dilations, strides
    });
    const memory = this.allocator.allocate(getSize(resultShape));

    if (compile) {
      const memories = this.getMemoryEntries(inputs);
      const memX = memories[0];
      const memW = memories[1];

      let info: ConvInfo = {
        shapeX: x.getShape(),
        widthX: memX.width,
        heightX: memX.height,
        shapeW: w.getShape(),
        widthW: memW.width,
        heightW: memW.height,
        shapeOutput: resultShape,
        widthOutput: memory.width,
        heightOutput: memory.height,

        pads, dilations, strides
      };

      if (bias !== undefined) {
        const memB = memories[2];

        info = {
          ...info,
          shapeB: bias.getShape(),
          heightB: memB.height,
          widthB: memB.width
        } as any;
      }

      this.operation.compile(info);

      this.compiled = true;
    }

    return {
      outputs: [new PrototypeTensor(resultShape, memory)]
    };
  }

  initializeForCompiling(inputs?: Tensor[]): void {
    if (inputs !== undefined) {
      if (inputs.length === 2) {
        this.operation = new ConvOperation(gpuConstructor, this.allocator);
      } else if(inputs.length === 3) {
        this.operation = new ConvBiasOperation(gpuConstructor, this.allocator);
      }
    }
  }
}