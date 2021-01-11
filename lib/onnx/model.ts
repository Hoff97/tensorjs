import Long from 'long';
import {onnx} from 'onnx-proto';
import { gl, glContext } from '../tensor/gpu/gl';
import Tensor from '../types';
import { toCPU, toGPU, toWASM } from '../util/convert';
import { TENSOR_INT64 } from './definitions';
import { OnnxNode } from './node';
import { nodeResolve } from './resolve';
import { Constants } from './types';
import { createTensor } from './util';

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
  private inputSet: Set<string> = new Set();
  private outputs: string[];

  private nodes: {[id: number]: OnnxNode} = {};
  private nodeIds: number[] = [];
  private defaultReady: number[] = [];

  private intermediaries: {[name: string]: Intermediary} = {};

  private constants: Constants = {};

  constructor(buffer: ArrayBuffer | Uint8Array) {
    let arr: Uint8Array;
    if (buffer instanceof ArrayBuffer) {
      arr = new Uint8Array(buffer);
    } else {
      arr = buffer;
    }
    const modelProto = onnx.ModelProto.decode(arr);

    let ver = modelProto.opsetImport[0].version;
    if (Long.isLong(ver)) {
      ver = (ver as Long).toNumber();
    }

    this.version = ver;

    this.inputs = modelProto.graph.input;
    for (let i = 0; i < this.inputs.length; i++) {
      this.inputSet.add(this.inputs[i].name);
    }
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

      if (node.variableInputs === 0) {
        this.defaultReady.push(i);
      }
    }
  }

  private initializer(initializer: onnx.ITensorProto[]) {
    for (let i = 0; i < initializer.length; i++) {
      const tensorProto = initializer[i];

      const tensor = createTensor(tensorProto);
      this.constants[tensorProto.name] = tensor;
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

    const nodesReady: number[] = [...this.defaultReady];

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

      const inputs: Tensor[] = [];
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
      glContext.flush();
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
        if (!this.inputSet.has(toDelete[i])) {
          const inter = intermediaryRes[toDelete[i]];
          inter.value.delete();
          delete intermediaryRes[toDelete[i]];
        }
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