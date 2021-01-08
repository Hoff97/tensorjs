import { AddNode } from "./nodes/add";
import { BatchNormNode } from "./nodes/batchNormalization";
import { ClipNode } from "./nodes/clip";
import { ConvNode } from "./nodes/conv";
import { GemmNode } from "./nodes/gemm";
import { ReduceMeanNode } from "./nodes/reduceMean";
import { NodeConstructor } from "./types";

export const nodeResolve: {[opType: string]: NodeConstructor} = {
  'Conv': (attributes, inputs, outputs, constants, onnxVersion) => new ConvNode(attributes, inputs, outputs, constants, onnxVersion),
  'BatchNormalization': (attributes, inputs, outputs, constants, onnxVersion) => new BatchNormNode(attributes, inputs, outputs, constants, onnxVersion),
  'Clip': (attributes, inputs, outputs, constants, onnxVersion) => new ClipNode(attributes, inputs, outputs, constants, onnxVersion),
  'Add': (attributes, inputs, outputs, constants, onnxVersion) => new AddNode(attributes, inputs, outputs, constants, onnxVersion),
  'ReduceMean': (attributes, inputs, outputs, constants, onnxVersion) => new ReduceMeanNode(attributes, inputs, outputs, constants, onnxVersion),
  'Gemm': (attributes, inputs, outputs, constants, onnxVersion) => new GemmNode(attributes, inputs, outputs, constants, onnxVersion)
};