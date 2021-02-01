import Tensor from '../../types';
import {NodeId} from '../model';
import {OnnxNode} from '../node';
import {Constants, OnnxModelI} from '../types';

export abstract class Optimization {
  /**
   * Finds possible places in the graph for application
   *
   * @param graph The graph to search for optimization applications
   *
   * @Returns a list of possible applications
   *          Each application consists of a list of nodes that will be replaced
   */
  abstract findApplications(model: OnnxModelI): NodeId[][];

  abstract apply(
    nodes: OnnxNode[],
    resolveConstant: (name: string) => Tensor | undefined,
    constants: Constants,
    onnxVersion: number
  ): OnnxNode;
}

export abstract class SequenceOptimization extends Optimization {
  protected nodeTypes: string[];

  constructor(nodeTypes: string[]) {
    super();
    this.nodeTypes = nodeTypes;
  }

  findApplications(model: OnnxModelI): NodeId[][] {
    const results: NodeId[][] = [];

    const nodes = model.getNodes();

    for (const nodeId in Object.keys(nodes)) {
      const node = nodes[nodeId];

      if (node !== undefined && node.getType() === this.nodeTypes[0]) {
        const app = this.checkApplication(model, nodeId);
        if (app !== undefined) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          results.push(app as any);
        }
      }
    }

    return results;
  }

  checkApplication(model: OnnxModelI, nodeId: string): string[] | undefined {
    const nodes = model.getNodes();

    const nodeSeq = [nodeId];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let lastNode = nodes[nodeId as any];
    const nodeInstances = [lastNode];
    for (let i = 1; i < this.nodeTypes.length; i++) {
      const nextNodeId = model.getNodeWithInput(lastNode.outputs[0]);
      if (nextNodeId !== undefined) {
        const nextNode = nodes[nextNodeId];

        if (nextNode.getType() === this.nodeTypes[i]) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          nodeSeq.push(nextNodeId as any);
          lastNode = nextNode;
          nodeInstances.push(lastNode);
        } else {
          return undefined;
        }
      } else {
        return undefined;
      }
    }
    if (this.canApply(nodeInstances)) {
      return nodeSeq;
    } else {
      return undefined;
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  canApply(nodes: OnnxNode[]) {
    return true;
  }
}
