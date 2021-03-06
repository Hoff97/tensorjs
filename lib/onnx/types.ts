import {onnx} from 'onnx-proto';
import {Mode} from '../model/module';
import Tensor from '../types';
import {OnnxNode} from './node';

export type Attributes = onnx.IAttributeProto[];
export type NodeConstructor = (
  attributes: Attributes,
  inputs: string[],
  outputs: string[],
  constants: Constants,
  onnxVersion: number,
  mode: Mode
) => OnnxNode;
export type Constants = {[name: string]: Tensor<any>};

export interface OnnxModelI {
  getNodes(): {[id: number]: OnnxNode};

  getNodeWithInput(output: string): number;
}
