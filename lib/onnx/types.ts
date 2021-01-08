import { onnx } from "onnx-proto";
import Tensor from "../types";
import { OnnxNode } from "./node";

export type Attributes = onnx.IAttributeProto[];
export type NodeConstructor = (attributes: Attributes,
                               inputs: string[],
                               outputs: string[],
                               constants: Constants,
                               onnxVersion: number) => OnnxNode;
export type Constants = {[name: string]: {
  type: string,
  value: Tensor
}};