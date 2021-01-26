import Long from 'long';
import {onnx} from 'onnx-proto';
import { PrototypeTensor } from '../tensor/cpu/prototype';
import { gl, glContext } from '../tensor/gpu/gl';
import { GPUMemoryAllocator } from '../tensor/gpu/memory';
import { GPUTensor } from '../tensor/gpu/tensor';
import Tensor from '../types';
import { toCPU, toGPU, toWASM } from '../util/convert';
import { TENSOR_INT64 } from './definitions';
import { OnnxNode } from './node';
import { ConcatNode } from './nodes/concat';
import { ConstantNode } from './nodes/constant';
import { nodeResolve } from './resolve';
import { Constants } from './types';
import { createTensor } from './util';

interface Intermediary {
  to: number[];
  deletable: boolean;
}

interface IntermediaryRes {
  value: Tensor;
  used: number;
}

interface ModelArgs {
  noConvertConstants?: string[];
  noConvertNodes?: number[];
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

  private noConvertConstants: Set<string>;
  private noConvertNodes: Set<number>;

  public allocator?: GPUMemoryAllocator;

  constructor(buffer: ArrayBuffer | Uint8Array, args?: ModelArgs) {
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

    if (args === undefined) {
      args = {};
    }

    this.noConvertConstants = new Set<string>(args.noConvertConstants !== undefined ? args.noConvertConstants : []);
    this.noConvertNodes = new Set<number>(args.noConvertNodes !== undefined ? args.noConvertNodes : []);
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

    const nodesReady: number[] = [...this.defaultReady];

    this.initializeForward(inputs, intermediaryRes, nodes, nodesReady);

    while(nodesReady.length > 0) {
      const nodeId = nodesReady.shift();
      const node = this.nodes[nodeId];

      const {inputs, toDelete} = this.getInputsToNode(node, intermediaryRes);

      let outputs: Tensor[];
      /*try {
        outputs = await node.forward(inputs);
      } catch (e) {
        console.error(`Error occurred in node ${nodeId} with inputs ${node.inputs} from nodes ${node.inputs.map(x => this.getNodeWithOutput(x))}`);
        throw e;
      }*/
      outputs = await node.forward(inputs);
      glContext.flush();

      this.propagateResults(node, intermediaryRes, outputs, nodes, nodesReady);

      for (let i = 0; i < toDelete.length; i++) {
        if (!this.inputSet.has(toDelete[i])) {
          const inter = intermediaryRes[toDelete[i]];
          if (this.allocator !== undefined && inter.value instanceof GPUTensor) {
            this.allocator.deallocate(inter.value.memory);
          } else {
            inter.value.delete();
          }
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
                              nodesReady: number[]) {
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
                             nodesReady: number[]) {
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

  public getNodeWithOutput(output: string) {
    for (let id of this.nodeIds) {
      if (this.nodes[id].outputs.findIndex(x => x === output) !== -1) {
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
        this.constants[i] = await toGPU(this.constants[i]);
      }
    }

    for (let i of this.nodeIds) {
      if (!this.noConvertNodes.has(i)) {
        await this.nodes[i].toGPU();
      }
    }
  }

  setAllocator(allocator: GPUMemoryAllocator) {
    this.allocator = allocator;
    for (let id of this.nodeIds) {
      this.nodes[id].setAllocator(this.allocator);
    }
  }

  async compileForInputs(inputs: Tensor[]) {
    if (this.allocator === undefined) {
      this.setAllocator(new GPUMemoryAllocator(gl));
    }

    for (let id of this.nodeIds) {
      this.nodes[id].initializeForCompiling();
    }

    this.staticForward(inputs, false);
    this.staticForward(inputs, false);

    this.staticForward(inputs, true);
  }

  async staticForward(inputs: Tensor[], compile: boolean) {
    const intermediaryRes: {[name: string]: IntermediaryRes} = {};

    const nodes: {[id: number]: {variableInputs: number}} = {};
    for (let i of this.nodeIds) {
      nodes[i] = {
        variableInputs: 0
      }
    }

    const nodesReady: number[] = [...this.defaultReady];

    this.initializeForward(inputs, intermediaryRes, nodes, nodesReady);

    while(nodesReady.length > 0) {
      const nodeId = nodesReady.shift();
      const node = this.nodes[nodeId];

      const {inputs, toDelete} = this.getInputsToNode(node, intermediaryRes);

      let { outputs } = await node.staticForward(inputs, compile);

      glContext.flush();

      this.propagateResults(node, intermediaryRes, outputs, nodes, nodesReady);

      for (let i = 0; i < toDelete.length; i++) {
        if (!this.inputSet.has(toDelete[i])) {
          const inter = intermediaryRes[toDelete[i]];

          if (inter.value instanceof PrototypeTensor) {
            this.allocator.deallocate(inter.value.memory);
          }

          delete intermediaryRes[toDelete[i]];
        }
      }
    }
  }
}