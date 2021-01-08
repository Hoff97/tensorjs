import { AddNode } from "./nodes/add";
import { BatchNormNode } from "./nodes/batchNormalization";
import { ClipNode } from "./nodes/clip";
import { ConcatNode } from "./nodes/concat";
import { ConstantNode } from "./nodes/constant";
import { ConvNode } from "./nodes/conv";
import { DivNode } from "./nodes/div";
import { ExpNode } from "./nodes/exp";
import { GemmNode } from "./nodes/gemm";
import { MatMulNode } from "./nodes/matMul";
import { MulNode } from "./nodes/mul";
import { ReduceMeanNode } from "./nodes/reduceMean";
import { ReduceSumNode } from "./nodes/reduceSum";
import { ReshapeNode } from "./nodes/reshape";
import { SubNode } from "./nodes/sub";
import { TileNode } from "./nodes/tile";
import { UnsqueezeNode } from "./nodes/unsqueeze";
import { NodeConstructor } from "./types";

export const nodeResolve: {[opType: string]: NodeConstructor} = {
  'Conv': (attributes, inputs, outputs, constants, onnxVersion) => new ConvNode(attributes, inputs, outputs, constants, onnxVersion),
  'BatchNormalization': (attributes, inputs, outputs, constants, onnxVersion) => new BatchNormNode(attributes, inputs, outputs, constants, onnxVersion),
  'Clip': (attributes, inputs, outputs, constants, onnxVersion) => new ClipNode(attributes, inputs, outputs, constants, onnxVersion),
  'Add': (attributes, inputs, outputs, constants, onnxVersion) => new AddNode(attributes, inputs, outputs, constants, onnxVersion),
  'ReduceMean': (attributes, inputs, outputs, constants, onnxVersion) => new ReduceMeanNode(attributes, inputs, outputs, constants, onnxVersion),
  'Gemm': (attributes, inputs, outputs, constants, onnxVersion) => new GemmNode(attributes, inputs, outputs, constants, onnxVersion),
  'Constant': (attributes, inputs, outputs, constants, onnxVersion) => new ConstantNode(attributes, inputs, outputs, constants, onnxVersion),
  'Reshape': (attributes, inputs, outputs, constants, onnxVersion) => new ReshapeNode(attributes, inputs, outputs, constants, onnxVersion),
  'Tile': (attributes, inputs, outputs, constants, onnxVersion) => new TileNode(attributes, inputs, outputs, constants, onnxVersion),
  'MatMul': (attributes, inputs, outputs, constants, onnxVersion) => new MatMulNode(attributes, inputs, outputs, constants, onnxVersion),
  'Exp': (attributes, inputs, outputs, constants, onnxVersion) => new ExpNode(attributes, inputs, outputs, constants, onnxVersion),
  'ReduceSum': (attributes, inputs, outputs, constants, onnxVersion) => new ReduceSumNode(attributes, inputs, outputs, constants, onnxVersion),
  'Sub': (attributes, inputs, outputs, constants, onnxVersion) => new SubNode(attributes, inputs, outputs, constants, onnxVersion),
  'Mul': (attributes, inputs, outputs, constants, onnxVersion) => new MulNode(attributes, inputs, outputs, constants, onnxVersion),
  'Div': (attributes, inputs, outputs, constants, onnxVersion) => new DivNode(attributes, inputs, outputs, constants, onnxVersion),
  'Unsqueeze': (attributes, inputs, outputs, constants, onnxVersion) => new UnsqueezeNode(attributes, inputs, outputs, constants, onnxVersion),
  'Concat': (attributes, inputs, outputs, constants, onnxVersion) => new ConcatNode(attributes, inputs, outputs, constants, onnxVersion)
};