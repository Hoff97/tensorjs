// eslint-disable-next-line node/no-extraneous-import
import Long from 'long';
import {onnx} from 'onnx-proto';
import {Variable} from '../autograd/variable';
import {Mode, Module} from '../model/module';
import {glContext} from '../tensor/gpu/gl';
import Tensor from '../types';
import {toCPU, toGPU, toWASM} from '../util/convert';
import {OnnxNode} from './node';
import {ConstantNode} from './nodes/constant';
import {defaultOptimizations} from './optimizations/default';
import {nodeResolve} from './resolve';
import {Constants} from './types';
import {createTensor} from './util';

export type NodeId = number;

interface Intermediary {
  to: NodeId[];
  deletable: boolean;
}

interface IntermediaryRes {
  value: Tensor<any>;
  used: number;
}

export interface ModelArgs {
  /**
   * Precision with which float tensors should be loaded. If 16 is specified,
   * all 32 bit floats are casted to 16 bit floats.
   *
   * Defaults to 32.
   */
  precision?: 16 | 32;
  /**
   * Constants that should not be transferred to another device.
   *
   * Useful for operations that should only happen only on the CPU
   */
  noConvertConstants?: string[];
  /**
   * Nodes that should not be transferred to another device
   *
   * Useful for operations that should only happen only on the CPU
   */
  noConvertNodes?: NodeId[];

  /**
   * If the model should be loaded in training or inference mode.
   * In inference mode, no gradients can be supported and
   * the model expects tensors as input.
   *
   * In training mode, gradients can be computed and
   * the inputs should be variables (although with noGrad: true if no gradient
   * is needed for the respective input)
   */
  mode?: Mode;
}

export class OnnxModel extends Module {
  private version: number;
  private inputSet: Set<string> = new Set();

  private nodes: {[id: number]: OnnxNode} = {};
  private nodeIds: NodeId[] = [];
  private defaultReady: NodeId[] = [];

  private intermediaries: {[name: string]: Intermediary} = {};

  private constants: Constants = {};

  private noConvertConstants: Set<string>;
  private noConvertNodes: Set<NodeId>;

  private modelProto: onnx.ModelProto;

  private nodeIdCounter = 10000;

  private precision: 32 | 16;

  public inputs: onnx.IValueInfoProto[];
  public outputs: string[];

  /**
   * Builds a new onnx model
   *
   * @param buffer Onnx model
   * @param args Optional arguments for the model
   */
  constructor(buffer: ArrayBuffer | Uint8Array, args?: ModelArgs) {
    super();
    if (args === undefined) {
      args = {};
    }

    this.noConvertConstants = new Set<string>(
      args.noConvertConstants !== undefined ? args.noConvertConstants : []
    );
    this.noConvertNodes = new Set<number>(
      args.noConvertNodes !== undefined ? args.noConvertNodes : []
    );

    this.mode = args.mode || 'inference';
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

    this.version = ver as number;

    //@ts-ignore
    this.inputs = this.modelProto.graph.input;
    for (let i = 0; i < this.inputs.length; i++) {
      this.inputSet.add(this.inputs[i].name as string);
    }
    //@ts-ignore
    this.outputs = this.modelProto.graph.output.map(x => x.name);

    //@ts-ignore
    this.initializer(this.modelProto.graph.initializer);

    this.initNodes(this.modelProto);
  }

  private initNodes(modelProto: onnx.ModelProto) {
    //@ts-ignore
    for (let i = 0; i < modelProto.graph.node.length; i++) {
      //@ts-ignore
      const nodeData = modelProto.graph.node[i];
      //@ts-ignore
      const cls = nodeResolve[nodeData.opType];

      if (cls === undefined) {
        throw new Error(`Node operator ${nodeData.opType} can not be resolved`);
      }

      const attributes = nodeData.attribute || [];
      const inputs = nodeData.input || [];
      const outputs = nodeData.output || [];

      const node = cls(
        attributes,
        inputs,
        outputs,
        this.constants,
        this.version,
        this.mode
      );
      this.nodes[i] = node;
      this.nodeIds.push(i);

      for (let j = 0; j < inputs.length; j++) {
        const input = inputs[j];
        if (this.intermediaries[input] === undefined) {
          this.intermediaries[input] = {
            to: [],
            deletable: true,
          };
        }
        this.intermediaries[input].to.push(i);
      }

      if (node.variableInputs === 0) {
        this.defaultReady.push(i);
      }

      if (nodeData.opType === 'Constant') {
        //@ts-ignore
        if (this.intermediaries[nodeData.output[0]] === undefined) {
          //@ts-ignore
          this.intermediaries[nodeData.output[0]] = {
            to: [],
            deletable: false,
          };
        } else {
          //@ts-ignore
          this.intermediaries[nodeData.output[0]].deletable = false;
        }
      }
    }

    for (const nodeId of this.nodeIds) {
      this.nodes[nodeId].initialize(name => this.resolveConstant(name));
    }
  }

  private initializer(initializer: onnx.ITensorProto[]) {
    for (let i = 0; i < initializer.length; i++) {
      const tensorProto = initializer[i];

      let tensor: Tensor<any> = createTensor(
        tensorProto,
        this.precision === 16
      );
      if (this.mode === 'train') {
        tensor = new Variable(tensor);
      }

      //@ts-ignore
      this.constants[tensorProto.name] = tensor;
    }
  }

  /**
   * Do a forward pass for the specified inputs
   *
   * @param wait Number of milliseconds to wait between each layer. This
   *             is especially useful, if your model is complex and
   *             you dont want your model to block your whole application.
   * @param returnIntermediary return after the given intermediary result
   *                           has been computed.
   */
  async forward(inputs: Tensor<any>[], wait?: number): Promise<Tensor<any>[]> {
    const intermediaryRes: {[name: string]: IntermediaryRes} = {};

    const nodes: {[id: number]: {variableInputs: number}} = {};
    for (const i of this.nodeIds) {
      nodes[i] = {
        variableInputs: 0,
      };
    }

    const nodesReady: NodeId[] = [...this.defaultReady];

    this.initializeForward(inputs, intermediaryRes, nodes, nodesReady);

    while (nodesReady.length > 0) {
      const nodeId = nodesReady.shift();
      //@ts-ignore
      const node = this.nodes[nodeId];

      const {inputs, toDelete} = this.getInputsToNode(node, intermediaryRes);

      let outputs: Tensor<any>[];
      try {
        outputs = await node.forward(inputs);
      } catch (e) {
        console.error(
          `Error occurred in node ${nodeId} with inputs ${
            node.inputs
          } from nodes ${node.inputs.map((x: string) =>
            this.getNodeWithOutput(x)
          )}`
        );
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
        await new Promise(resolve => {
          setTimeout(resolve, wait);
        });
      }
    }

    const outputs: Tensor<any>[] = [];
    for (let i = 0; i < this.outputs.length; i++) {
      outputs.push(intermediaryRes[this.outputs[i]].value);
    }

    return outputs;
  }

  protected initializeForward(
    inputs: Tensor<any>[],
    intermediaryRes: {[name: string]: IntermediaryRes},
    nodes: {[id: number]: {variableInputs: number}},
    nodesReady: NodeId[]
  ) {
    for (let i = 0; i < inputs.length; i++) {
      //@ts-ignore
      intermediaryRes[this.inputs[i].name] = {
        value: inputs[i],
        used: 0,
      };

      //@ts-ignore
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

  protected getInputsToNode(
    node: OnnxNode,
    intermediaryRes: {[name: string]: IntermediaryRes}
  ) {
    const inputs: Tensor<any>[] = [];
    const toDelete: string[] = [];
    for (let i = 0; i < node.inputs.length; i++) {
      const input = node.inputs[i];
      if (this.constants[input] !== undefined) {
        inputs.push(this.constants[input]);
      } else {
        const inter = intermediaryRes[input];
        inter.used++;
        if (
          inter.used >= this.intermediaries[input].to.length &&
          this.intermediaries[input].deletable
        ) {
          toDelete.push(input);
        }
        inputs.push(inter.value);
      }
    }

    return {inputs, toDelete};
  }

  protected propagateResults(
    node: OnnxNode,
    intermediaryRes: {[name: string]: IntermediaryRes},
    outputs: Tensor<any>[],
    nodes: {[id: number]: {variableInputs: number}},
    nodesReady: NodeId[]
  ) {
    for (let i = 0; i < node.outputs.length; i++) {
      const output = node.outputs[i];
      intermediaryRes[output] = {
        value: outputs[i],
        used: 0,
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

  /**
   * Transfer the model to the CPU
   */
  async toCPU() {
    for (const i in this.constants) {
      if (!this.noConvertConstants.has(i)) {
        this.constants[i] = await toCPU(this.constants[i]);
      }
    }

    for (const i of this.nodeIds) {
      if (!this.noConvertNodes.has(i)) {
        await this.nodes[i].toCPU();
      }
    }
  }

  /**
   * Transfer the model to WASM
   */
  async toWASM() {
    for (const i in this.constants) {
      if (!this.noConvertConstants.has(i)) {
        this.constants[i] = await toWASM(this.constants[i]);
      }
    }

    for (const i of this.nodeIds) {
      if (!this.noConvertNodes.has(i)) {
        await this.nodes[i].toWASM();
      }
    }
  }

  /**
   * Transfer the model to the GPU
   */
  async toGPU() {
    for (const i in this.constants) {
      if (!this.noConvertConstants.has(i)) {
        this.constants[i] = await toGPU(this.constants[i]);
      }
    }

    for (const i of this.nodeIds) {
      if (!this.noConvertNodes.has(i)) {
        await this.nodes[i].toGPU();
      }
    }
  }

  /**
   * Optimize the model.
   */
  public optimize() {
    for (const optimization of defaultOptimizations) {
      //@ts-ignore
      const applications = optimization.findApplications(this);

      for (const nodeIds of applications) {
        const nodes = nodeIds.map(x => this.nodes[x]);
        const newNode = optimization.apply(
          nodes,
          name => this.resolveConstant(name),
          this.constants,
          this.version
        );

        const outputs = new Set(newNode.outputs);

        for (const nodeId of nodeIds) {
          this.removeNode(nodeId, outputs);
        }

        this.insertNode(newNode);
      }
    }

    this.prune();
  }

  public prune(intermediariesToDelete?: string[]) {
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const nodesToDelete = this.pruneIntermediaries(intermediariesToDelete);

      intermediariesToDelete = [];

      if (nodesToDelete.size > 0) {
        nodesToDelete.forEach(id => {
          const interToDelete = this.removeNode(id, new Set());
          intermediariesToDelete = intermediariesToDelete?.concat(
            interToDelete
          );
        });
      } else {
        break;
      }
    }
  }

  private pruneIntermediaries(intermediariesToDelete?: string[]) {
    const nodesToDelete = new Set<number>();

    if (intermediariesToDelete === undefined) {
      intermediariesToDelete = [];
    }

    for (let i = 0; i < intermediariesToDelete.length; i++) {
      const id = intermediariesToDelete[i];
      const nodeOutputId = this.getNodeWithOutput(id);
      if (nodeOutputId !== undefined) {
        nodesToDelete.add(nodeOutputId);
      }
      const nodeInputId = this.getNodeWithInput(id);
      if (nodeInputId !== undefined) {
        nodesToDelete.add(nodeInputId);
      }
    }

    for (const id in this.intermediaries) {
      const intermediary = this.intermediaries[id];
      if (
        intermediary.to.length === 0 &&
        this.outputs.find(x => x === id) === undefined
      ) {
        intermediariesToDelete.push(id);
        const nodeOutputId = this.getNodeWithOutput(id);
        if (nodeOutputId !== undefined) {
          nodesToDelete.add(nodeOutputId);
        }
        const nodeInputId = this.getNodeWithInput(id);
        if (nodeInputId !== undefined) {
          nodesToDelete.add(nodeInputId);
        }
      }
    }

    for (const id of intermediariesToDelete) {
      delete this.intermediaries[id];
    }

    return nodesToDelete;
  }

  private removeNode(
    nodeId: number,
    preserveIntermediaries: Set<string>
  ): string[] {
    const node = this.nodes[nodeId];
    for (const input of node.inputs) {
      if (this.intermediaries[input] !== undefined) {
        this.intermediaries[input].to = this.intermediaries[input].to.filter(
          x => x.toString() !== nodeId.toString()
        );
      }
    }

    const intermediariesToDelete = [];
    if (!preserveIntermediaries.has(node.outputs[0])) {
      intermediariesToDelete.push(node.outputs[0]);
    }

    this.nodeIds = this.nodeIds.filter(x => x.toString() !== nodeId.toString());
    this.nodes[nodeId].delete();
    delete this.nodes[nodeId];

    this.defaultReady = this.defaultReady.filter(x => x !== nodeId);

    return intermediariesToDelete;
  }

  private insertNode(node: OnnxNode) {
    const id = this.nodeIdCounter++;

    this.nodeIds.push(id);
    this.nodes[id] = node;

    for (const input of node.inputs) {
      this.intermediaries[input].to.push(id);
    }
  }

  // Utility functions

  public getNodeWithOutput(output: string) {
    for (const id of this.nodeIds) {
      if (this.nodes[id].outputs.findIndex(x => x === output) !== -1) {
        return id;
      }
    }
    return undefined;
  }

  public getNodeWithInput(output: string) {
    for (const id of this.nodeIds) {
      if (this.nodes[id].inputs.findIndex(x => x === output) !== -1) {
        return id;
      }
    }
    return undefined;
  }

  public resolveConstant(name: string) {
    if (this.constants[name] !== undefined) {
      return this.constants[name];
    }
    const nodeIdOut = this.getNodeWithOutput(name);
    //@ts-ignore
    const nodeOut = this.nodes[nodeIdOut];
    if (nodeOut instanceof ConstantNode) {
      return nodeOut.tensor;
    }
    return undefined;
  }

  public getNodes() {
    return this.nodes;
  }

  /**
   * Deletes the model
   *
   * This will release the memory/framebuffers (depending on the backend)
   */
  public delete() {
    for (const c in this.constants) {
      this.constants[c].delete();
    }

    for (const nodeId of this.nodeIds) {
      this.nodes[nodeId].delete();
    }
  }

  getSubModules(): Module[] {
    const modules: Module[] = super.getSubModules();
    for (const nodeId of this.nodeIds) {
      modules.push(this.nodes[nodeId]);
    }
    return modules;
  }

  getParameters(): Variable<any>[] {
    const parameters: Variable<any>[] = super.getParameters();
    for (const c in this.constants) {
      if (this.constants[c] instanceof Variable) {
        parameters.push(this.constants[c] as Variable<any>);
      }
    }
    return parameters;
  }
}
