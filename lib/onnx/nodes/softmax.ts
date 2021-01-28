import { BinaryOpInfo } from "../../ops/gpu/binaryOperation";
import { DivideOperation } from "../../ops/gpu/divide";
import { ExpOperation } from "../../ops/gpu/exp";
import { MaxOperation } from "../../ops/gpu/max";
import { PoolInfo } from "../../ops/gpu/pool";
import { SubtractOperation } from "../../ops/gpu/subtract";
import { SumOperation } from "../../ops/gpu/sum";
import { UnaryOpInfo } from "../../ops/gpu/unaryOperation";
import { PrototypeTensor } from "../../tensor/cpu/prototype";
import { CPUTensor } from "../../tensor/cpu/tensor";
import { gpuConstructor, GPUTensor } from "../../tensor/gpu/tensor";
import Tensor, { Precision } from "../../types";
import { OnnxNode } from "../node";
import { Attributes, Constants } from "../types";

export class SoftmaxNode extends OnnxNode {
  private axis?: number;

  private compiled = false;
  private maxOperation?: MaxOperation<GPUTensor>;
  private subtractOperation?: SubtractOperation<GPUTensor>;
  private expOperation?: ExpOperation<GPUTensor>;
  private sumOperation?: SumOperation<GPUTensor>;
  private divideOperation?: DivideOperation<GPUTensor>;

  constructor(attributes: Attributes, inputs: string[], outputs: string[], constants: Constants, onnxVersion: number) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.axis = this.getAttributeInt("axis");
  }

  async defaultForward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    const shapeX = x.getShape();

    let ax = this.axis;
    if (ax === undefined) {
      if (this.onnxVersion < 13) {
        ax = 1;
      } else {
        ax = shapeX.length - 1;
      }
    }

    const sh1 = shapeX.slice(0, ax).reduce((x,y) => x*y, 1);

    const reshaped = x.reshape([sh1,-1], false);

    const max = reshaped.max(1, true);
    const normalized = reshaped.subtract(max);
    const exp = normalized.exp();
    const sum = exp.sum(1, true);
    const result = exp.divide(sum);

    max.delete();
    normalized.delete();
    exp.delete();
    sum.delete();

    return [result.reshape(shapeX, false)];
  }

  async compiledForward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    const shapeX = x.getShape();

    let ax = this.axis;
    if (ax === undefined) {
      if (this.onnxVersion < 13) {
        ax = 1;
      } else {
        ax = shapeX.length - 1;
      }
    }

    const sh1 = shapeX.slice(0, ax).reduce((x,y) => x*y, 1);

    const reshaped = x.reshape([sh1,-1], false);

    const max = this.maxOperation.calc({X: reshaped as GPUTensor, axes: [1], keepDims: true});
    const normalized = this.subtractOperation.calc({A: reshaped as GPUTensor, B: max, outputShape: reshaped.getShape()});
    const exp = this.expOperation.calc({input: normalized});
    const sum = this.sumOperation.calc({X: exp, axes: [1], keepDims: true});
    const result = this.divideOperation.calc({A: exp as GPUTensor, B: sum, outputShape: reshaped.getShape()});

    max.delete(this.allocator);
    normalized.delete(this.allocator);
    exp.delete(this.allocator);
    sum.delete(this.allocator);

    return [result.reshape(shapeX, false)];
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (!this.compiled) {
      return this.defaultForward(inputs);
    } else {
      return this.compiledForward(inputs);
    }
  }

  async staticForward(inputs: Tensor[], compile: boolean, precision: Precision): Promise<{ outputs: (CPUTensor | PrototypeTensor)[]; }> {
    if (this.allStaticCPU(inputs)) {
      return this.defaultStaticForward(inputs);
    }

    const x = inputs[0];
    const shapeX = x.getShape();

    let ax = this.axis;
    if (ax === undefined) {
      if (this.onnxVersion < 13) {
        ax = 1;
      } else {
        ax = shapeX.length - 1;
      }
    }
    const sh1 = shapeX.slice(0, ax).reduce((x,y) => x*y, 1);
    const sh2 = shapeX.slice(ax).reduce((x,y) => x*y, 1);
    const newShape = [sh1, sh2];

    const maxMemory = this.allocator.allocate(sh1, precision);
    const normalizedMemory = this.allocator.allocate(sh1*sh2, precision);
    const expMemory = this.allocator.allocate(sh1*sh2, precision);
    const sumMemory = this.allocator.allocate(sh1, precision);
    const memory = this.allocator.allocate(sh1*sh2, precision);

    if (compile) {
      const [xMem] = this.getMemoryEntries(inputs);

      const infoMax: PoolInfo = {
        shapeX: [sh1, sh2],
        widthX: xMem.width,
        heightX: xMem.height,

        shapeOutput: [sh1, 1],
        widthOutput: maxMemory.width,
        heightOutput: maxMemory.height,

        axes: [1], keepDims: true
      };
      this.maxOperation.compile(infoMax, precision);

      const infoSub: BinaryOpInfo = {
        shapeA: [sh1, sh2],
        widthA: xMem.width,
        heightA: xMem.height,

        shapeB: [sh1, 1],
        widthB: maxMemory.width,
        heightB: maxMemory.height,

        shapeOutput: [sh1, sh2],
        widthOutput: normalizedMemory.width,
        heightOutput: normalizedMemory.height,
      };
      this.subtractOperation.compile(infoSub, precision);

      const infoExp: UnaryOpInfo = {
        shapeX: [sh1, sh2],
        widthX: normalizedMemory.width,
        heightX: normalizedMemory.height,

        shapeOutput: [sh1, sh2],
        widthOutput: expMemory.width,
        heightOutput: expMemory.height
      };
      this.expOperation.compile(infoExp, precision);

      const infoSum: PoolInfo = {
        shapeX: [sh1, sh2],
        widthX: expMemory.width,
        heightX: expMemory.height,

        shapeOutput: [sh1, 1],
        widthOutput: sumMemory.width,
        heightOutput: sumMemory.height,

        axes: [1], keepDims: true
      };
      this.sumOperation.compile(infoSum, precision);

      const infoDivide: BinaryOpInfo = {
        shapeA: [sh1, sh2],
        widthA: xMem.width,
        heightA: xMem.height,

        shapeB: [sh1, 1],
        widthB: sumMemory.width,
        heightB: sumMemory.height,

        shapeOutput: [sh1, sh2],
        widthOutput: memory.width,
        heightOutput: memory.height,
      };
      this.divideOperation.compile(infoDivide, precision);

      this.compiled = true;
    }

    this.allocator.deallocate(maxMemory);
    this.allocator.deallocate(normalizedMemory);
    this.allocator.deallocate(expMemory);
    this.allocator.deallocate(sumMemory);

    return {
      outputs: [new PrototypeTensor(x.getShape(), memory)]
    };
  }

  initializeForCompiling(): void {
    this.maxOperation = new MaxOperation(gpuConstructor, this.allocator);
    this.subtractOperation = new SubtractOperation(gpuConstructor, this.allocator);
    this.expOperation = new ExpOperation(gpuConstructor, this.allocator);
    this.sumOperation = new SumOperation(gpuConstructor, this.allocator);
    this.divideOperation = new DivideOperation(gpuConstructor, this.allocator);
  }

  getType() {
    return 'Softmax';
  }
}