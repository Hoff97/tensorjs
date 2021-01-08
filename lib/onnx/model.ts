import Long from 'long';
import {onnx} from 'onnx-proto';
import CPUTensor from '../tensor/cpu/tensor';
import Tensor from '../types';
import { toCPU, toGPU, toWASM } from '../util/convert';
import { TENSOR_FLOAT, TENSOR_INT64 } from './definitions';
import { OnnxNode } from './node';
import { nodeResolve } from './resolve';
import { Constants } from './types';

interface Intermediary {
  to: number[];
}

interface IntermediaryRes {
  value: Tensor;
  used: number;
}

export class OnnxModel {
  private version: number;
  public inputs: onnx.IValueInfoProto[];
  private outputs: string[];

  private nodes: {[id: number]: OnnxNode} = {};
  private nodeIds: number[] = [];

  private intermediaries: {[name: string]: Intermediary} = {};

  private constants: Constants = {};

  constructor(buffer: ArrayBuffer) {
    const arr = new Uint8Array(buffer);
    const modelProto = onnx.ModelProto.decode(arr);

    let ver = modelProto.opsetImport[0].version;
    if (Long.isLong(ver)) {
      ver = (ver as Long).toNumber();
    }

    this.version = ver;

    this.inputs = modelProto.graph.input;
    this.outputs = modelProto.graph.output.map(x => x.name);

    this.initializer(modelProto.graph.initializer);

    this.initNodes(modelProto);
  }

  private initNodes(modelProto: onnx.ModelProto) {
    for (let i = 0; i < modelProto.graph.node.length; i++) {
      const nodeData = modelProto.graph.node[i];
      const cls = nodeResolve[nodeData.opType];

      if (cls === undefined) {
        throw new Error(`Node operator ${nodeData.opType} can not be resolved`);
      }

      const attributes = nodeData.attribute || [];
      const inputs = nodeData.input || [];
      const outputs = nodeData.output || [];

      const node = cls(attributes, inputs, outputs, this.constants, this.version);
      this.nodes[i] = node;
      this.nodeIds.push(i);

      for (let j = 0; j < inputs.length; j++) {
        const input = inputs[j];
        if (this.intermediaries[input] === undefined) {
          this.intermediaries[input] = {
            to: []
          };
        }
        this.intermediaries[input].to.push(i);
      }
    }
  }

  private initializer(initializer: onnx.ITensorProto[]) {
    for (let i = 0; i < initializer.length; i++) {
      const tensorProto = initializer[i];

      const tensor = this.createTensor(tensorProto);
      this.constants[tensorProto.name] = tensor;
    }
  }

  private createTensor(tensorProto: onnx.ITensorProto): Tensor {
    if (tensorProto.segment !== undefined && tensorProto.segment !== null) {
      throw new Error('Handling of tensor proto segment not yet implemented');
    }

    let shape: number[] = tensorProto.dims as number[];
    if (shape === undefined || shape === null) {
      throw new Error('Tensor shape must be specified');
    }
    for (let i = 0; i < shape.length; i++) {
      if (Long.isLong(shape[i])) {
        shape[i] = (shape[i] as any).toNumber();
      }
    }
    if (shape.length === 0) {
      shape = [1];
    }

    if (tensorProto.dataType === TENSOR_FLOAT) {
      if (tensorProto.floatData && tensorProto.floatData.length > 0) {
        return new CPUTensor(shape, tensorProto.floatData);
      } else if (tensorProto.rawData && tensorProto.rawData.length > 0) {
        const buffer = tensorProto.rawData.buffer.slice(tensorProto.rawData.byteOffset, tensorProto.rawData.byteOffset+tensorProto.rawData.byteLength);
        const values = new Float32Array(buffer);
        return new CPUTensor(shape, values);
      } else {
        throw new Error('Cant process float tensor without float or raw data');
      }
    } else if (tensorProto.dataType === TENSOR_INT64) {
      if (tensorProto.rawData && tensorProto.rawData.length > 0) {
        if (tensorProto.rawData.length > 8) {
          throw new Error('Can only process single int64 values');
        }
        const value = Long.fromBytes(Array.from(tensorProto.rawData)).toNumber();
        return new CPUTensor(shape, [value]);
      } else {
        throw new Error('Cant process int64 tensor without raw data');
      }
    } else {
      throw new Error(`Handling of tensor type ${tensorProto.dataType} not yet implemented`);
    }
  }

  forward(inputs: Tensor[]): Tensor[] {
    const intermediaryRes: {[name: string]: IntermediaryRes} = {};

    const nodes: {[id: number]: {variableInputs: number}} = {};
    for (let i of this.nodeIds) {
      nodes[i] = {
        variableInputs: 0
      }
    }

    const nodesReady: number[] = [];

    for (let i = 0; i < inputs.length; i++) {
      intermediaryRes[this.inputs[i].name] = {
        value: inputs[i],
        used: 0
      };

      const inter = this.intermediaries[this.inputs[i].name];
      for (let j = 0; j < inter.to.length; j++) {
        const id = inter.to[j];
        nodes[id].variableInputs++;

        if (nodes[id].variableInputs === this.nodes[id].variableInputs) {
          nodesReady.push(id);
          delete nodes[id];
        }
      }
    }

    while(nodesReady.length > 0) {
      const toDelete: string[] = [];

      const nodeId = nodesReady.shift();
      const node = this.nodes[nodeId];

      const inputs = [];
      for (let i = 0; i < node.inputs.length; i++) {
        const input = node.inputs[i];
        if (this.constants[input] !== undefined) {
          inputs.push(this.constants[input]);
        } else {
          const inter = intermediaryRes[input];
          inter.used++;
          if (inter.used === this.intermediaries[input].to.length) {
            toDelete.push(input);
          }
          inputs.push(inter.value);
        }
      }

      const outputs = node.forward(inputs);
      for (let i = 0; i < node.outputs.length; i++) {
        const output = node.outputs[i];
        intermediaryRes[output] = {
          value: outputs[i],
          used: 0
        };

        const inter = this.intermediaries[output];

        if (inter !== undefined) {
          for (let j = 0; j < inter.to.length; j++) {
            const id = inter.to[j];
            nodes[id].variableInputs++;

            if (nodes[id].variableInputs === this.nodes[id].variableInputs) {
              nodesReady.push(id);
              delete nodes[id];
            }
          }
        }
      }

      for (let i = 0; i < toDelete.length; i++) {
        const inter = intermediaryRes[toDelete[i]];
        inter.value.delete();
        delete intermediaryRes[toDelete[i]];
      }
    }

    const outputs: Tensor[] = [];
    for (let i = 0; i < this.outputs.length; i++) {
      outputs.push(intermediaryRes[this.outputs[i]].value);
    }

    return outputs;
  }

  async toCPU() {
    for (let i in this.constants) {
      this.constants[i] = await toCPU(this.constants[i]);
    }

    for (let i of this.nodeIds) {
      await this.nodes[i].toCPU();
    }
  }

  async toWASM() {
    for (let i in this.constants) {
      this.constants[i] = await toWASM(this.constants[i]);
    }

    for (let i of this.nodeIds) {
      await this.nodes[i].toWASM();
    }
  }

  async toGPU() {
    for (let i in this.constants) {
      this.constants[i] = await toGPU(this.constants[i]);
    }

    for (let i of this.nodeIds) {
      await this.nodes[i].toGPU();
    }
  }
}