import Long from 'long';
import {onnx} from 'onnx-proto';
import { PrototypeTensor } from '../tensor/cpu/prototype';
import { gl, glContext } from '../tensor/gpu/gl';
import { GPUMemoryAllocator } from '../tensor/gpu/memory';
import { GPUTensor } from '../tensor/gpu/tensor';
import Tensor, { Precision } from '../types';
import { toCPU, toGPU, toWASM } from '../util/convert';
import { Dict } from '../util/datastructs/dict';
import { OnnxNode } from './node';
import { ConstantNode } from './nodes/constant';
import { defaultOptimizations } from './optimizations/default';
import { nodeResolve } from './resolve';
import { Constants } from './types';
import { createTensor } from './util';

export type NodeId = number;

interface Intermediary {
  to: NodeId[];
  deletable: boolean;
}

interface IntermediaryRes {
  value: Tensor;
  used: number;
}

interface ModelArgs {
  noConvertConstants?: string[];
  noConvertNodes?: NodeId[];
  precision?: Precision;
}

export class OnnxModel {
  private version: number;
  private inputSet: Set<string> = new Set();
  private outputs: string[];

  private nodes: {[id: number]: OnnxNode} = {};
  private nodeIds: NodeId[] = [];
  private defaultReady: NodeId[] = [];

  private intermediaries: {[name: string]: Intermediary} = {};

  private constants: Constants = {};

  private noConvertConstants: Set<string>;
  private noConvertNodes: Set<NodeId>;

  private modelProto: onnx.ModelProto;

  private nodeIdCounter = 10000;

  private precision: Precision;

  public inputs: onnx.IValueInfoProto[];

  constructor(buffer: ArrayBuffer | Uint8Array, args?: ModelArgs) {
    if (args === undefined) {
      args = {};
    }

    this.noConvertConstants = new Set<string>(args.noConvertConstants !== undefined ? args.noConvertConstants : []);
    this.noConvertNodes = new Set<number>(args.noConvertNodes !== undefined ? args.noConvertNodes : []);

    this.precision = args.precision || 32;

    let arr: Uint8Array;
    if (buffer instanceof ArrayBuffer) {
      arr = new Uint8Array(buffer);
    } else {
      arr = buffer;
    }
    this.modelProto = onnx.ModelProto.decode(arr);

    let ver = this.modelProto.opsetImport[0].version;
    if (Long.isLong(ver)) {
      ver = (ver as Long).toNumber();
    }

    this.version = ver;

    this.inputs = this.modelProto.graph.input;
    for (let i = 0; i < this.inputs.length; i++) {
      this.inputSet.add(this.inputs[i].name);
    }
    this.outputs = this.modelProto.graph.output.map(x => x.name);

    this.initializer(this.modelProto.graph.initializer);

    this.initNodes(this.modelProto);
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
            to: [],
            deletable: true
          };
        }
        this.intermediaries[input].to.push(i);
      }

      if (node.variableInputs === 0) {
        this.defaultReady.push(i);
      }

      if (nodeData.opType === 'Constant') {
        if (this.intermediaries[nodeData.output[0]] === undefined) {
          this.intermediaries[nodeData.output[0]] = {
            to: [],
            deletable: false
          }
        } else {
          this.intermediaries[nodeData.output[0]].deletable = false;
        }
      }
    }

    for (let nodeId of this.nodeIds) {
      this.nodes[nodeId].initialize((name) => this.resolveConstant(name));
    }
  }

  private initializer(initializer: onnx.ITensorProto[]) {
    for (let i = 0; i < initializer.length; i++) {
      const tensorProto = initializer[i];

      const tensor = createTensor(tensorProto);
      this.constants[tensorProto.name] = tensor;
    }
  }

  async forward(inputs: Tensor[], wait?: number): Promise<Tensor[]> {
    const intermediaryRes: {[name: string]: IntermediaryRes} = {};

    const nodes: {[id: number]: {variableInputs: number}} = {};
    for (let i of this.nodeIds) {
      nodes[i] = {
        variableInputs: 0
      }
    }

    const nodesReady: NodeId[] = [...this.defaultReady];

    this.initializeForward(inputs, intermediaryRes, nodes, nodesReady);

    while(nodesReady.length > 0) {
      const nodeId = nodesReady.shift();
      const node = this.nodes[nodeId];

      const {inputs, toDelete} = this.getInputsToNode(node, intermediaryRes);

      let outputs: Tensor[];
      try {
        outputs = await node.forward(inputs);
      } catch (e) {
        console.error(`Error occurred in node ${nodeId} with inputs ${node.inputs} from nodes ${node.inputs.map(x => this.getNodeWithOutput(x))}`);
        throw e;
      }
      glContext.flush();

      this.propagateResults(node, intermediaryRes, outputs, nodes, nodesReady);

      for (let i = 0; i < toDelete.length; i++) {
        if (!this.inputSet.has(toDelete[i])) {
          const inter = intermediaryRes[toDelete[i]];
          inter.value.delete();
          delete intermediaryRes[toDelete[i]];
        }
      }

      if (wait !== undefined) {
        await new Promise((resolve, _) => {
          setTimeout(resolve, wait);
        });
      }
    }

    const outputs: Tensor[] = [];
    for (let i = 0; i < this.outputs.length; i++) {
      outputs.push(intermediaryRes[this.outputs[i]].value);
    }

    return outputs;
  }

  protected initializeForward(inputs: Tensor[], intermediaryRes: {[name: string]: IntermediaryRes},
                              nodes: {[id: number]: {variableInputs: number}},
                              nodesReady: NodeId[]) {
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
  }

  protected getInputsToNode(node: OnnxNode, intermediaryRes: {[name: string]: IntermediaryRes}) {
    const inputs: Tensor[] = [];
    const toDelete: string[] = [];
    for (let i = 0; i < node.inputs.length; i++) {
      const input = node.inputs[i];
      if (this.constants[input] !== undefined) {
        inputs.push(this.constants[input]);
      } else {
        const inter = intermediaryRes[input];
        inter.used++;
        if (inter.used >= this.intermediaries[input].to.length && this.intermediaries[input].deletable) {
          toDelete.push(input);
        }
        inputs.push(inter.value);
      }
    }

    return {inputs, toDelete};
  }

  protected propagateResults(node: OnnxNode, intermediaryRes: {[name: string]: IntermediaryRes},
                             outputs: Tensor[], nodes: {[id: number]: {variableInputs: number}},
                             nodesReady: NodeId[]) {
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
  }

  async toCPU() {
    for (let i in this.constants) {
      if (!this.noConvertConstants.has(i)) {
        this.constants[i] = await toCPU(this.constants[i]);
      }
    }

    for (let i of this.nodeIds) {
      if (!this.noConvertNodes.has(i)) {
        await this.nodes[i].toCPU();
      }
    }
  }

  async toWASM() {
    for (let i in this.constants) {
      if (!this.noConvertConstants.has(i)) {
        this.constants[i] = await toWASM(this.constants[i]);
      }
    }

    for (let i of this.nodeIds) {
      if (!this.noConvertNodes.has(i)) {
        await this.nodes[i].toWASM();
      }
    }
  }

  async toGPU() {
    for (let i in this.constants) {
      if (!this.noConvertConstants.has(i)) {
        this.constants[i] = await toGPU(this.constants[i], this.precision);
      }
    }

    for (let i of this.nodeIds) {
      if (!this.noConvertNodes.has(i)) {
        await this.nodes[i].toGPU(this.precision);
      }
    }
  }

  public optimize() {
    for (let optimization of defaultOptimizations) {
      const applications = optimization.findApplications(this);

      for (let nodeIds of applications) {
        const nodes = nodeIds.map(x => this.nodes[x]);
        const newNode = optimization.apply(nodes, (name) => this.resolveConstant(name), this.constants, this.version);

        const outputs = new Set(newNode.outputs);

        for (let nodeId of nodeIds) {
          this.removeNode(nodeId, outputs);
        }

        this.insertNode(newNode);
      }
    }

    this.prune();
  }

  private prune() {
    while (true) {
      const nodesToDelete = this.pruneIntermediaries();

      if (nodesToDelete.size > 0) {
        nodesToDelete.forEach(id => {
          this.removeNode(id, new Set());
        });
      } else {
        break;
      }
    }
  }

  private pruneIntermediaries() {
    const nodesToDelete = new Set<number>();

    const intermediariesToDelete: string[] = [];

    for (let id in this.intermediaries) {
      const intermediary = this.intermediaries[id];
      if (intermediary.to.length === 0) {
        intermediariesToDelete.push(id);
        const nodeId = this.getNodeWithOutput(id);
        if (nodeId !== undefined) {
          nodesToDelete.add(nodeId);
        }
      }
    }

    for (let id of intermediariesToDelete) {
      delete this.intermediaries[id];
    }

    return nodesToDelete;
  }

  private removeNode(nodeId: number, preserveIntermediaries: Set<string>) {
    const node = this.nodes[nodeId];
    for (let input of node.inputs) {
      if (this.intermediaries[input] !== undefined) {
        this.intermediaries[input].to = this.intermediaries[input].to.filter(x => x.toString() !== nodeId.toString());
      }
    }
    if (!preserveIntermediaries.has(node.outputs[0])) {
      delete this.intermediaries[node.outputs[0]];
    }

    this.nodeIds = this.nodeIds.filter(x => x.toString() !== nodeId.toString());
    this.nodes[nodeId].delete();
    delete this.nodes[nodeId];
  }

  private insertNode(node: OnnxNode) {
    const id = this.nodeIdCounter++;

    this.nodeIds.push(id);
    this.nodes[id] = node;

    for (let input of node.inputs) {
      this.intermediaries[input].to.push(id);
    }
  }

  // Utility functions

  public getNodeWithOutput(output: string) {
    for (let id of this.nodeIds) {
      if (this.nodes[id].outputs.findIndex(x => x === output) !== -1) {
        return id;
      }
    }
  }

  public getNodeWithInput(output: string) {
    for (let id of this.nodeIds) {
      if (this.nodes[id].inputs.findIndex(x => x === output) !== -1) {
        return id;
      }
    }
  }

  public resolveConstant(name: string) {
    if (this.constants[name] !== undefined) {
      return this.constants[name];
    }
    const nodeIdOut = this.getNodeWithOutput(name);
    const nodeOut = this.nodes[nodeIdOut];
    if (nodeOut instanceof ConstantNode) {
      return nodeOut.tensor;
    }
    return undefined;
  }

  public getNodes() {
    return this.nodes;
  }

  public delete() {
    for (let c in this.constants) {
      this.constants[c].delete();
    }

    for (let nodeId of this.nodeIds) {
      this.nodes[nodeId].delete();
    }
  }
}