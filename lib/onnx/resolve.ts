import { AddNode } from "./nodes/binary/add";
import { BatchNormalizationNode } from "./nodes/batchNormalization";
import { CastNode } from "./nodes/cast";
import { CeilNode } from "./nodes/unary/ceil";
import { ClipNode } from "./nodes/clip";
import { ConcatNode } from "./nodes/concat";
import { ConstantNode } from "./nodes/constant";
import { ConstantOfShapeNode } from "./nodes/constantOfShape";
import { ConvNode } from "./nodes/conv";
import { DivNode } from "./nodes/binary/div";
import { ExpNode } from "./nodes/unary/exp";
import { ExpandNode } from "./nodes/expand";
import { FloorNode } from "./nodes/unary/floor";
import { GatherNode } from "./nodes/gather";
import { GemmNode } from "./nodes/gemm";
import { InstanceNormalizationNode } from "./nodes/instanceNormalization";
import { MatMulNode } from "./nodes/matMul";
import { MulNode } from "./nodes/binary/mul";
import { PadNode } from "./nodes/pad";
import { ReduceMaxNode } from "./nodes/reduce/reduceMax";
import { ReduceMeanNode } from "./nodes/reduce/reduceMean";
import { ReduceSumNode } from "./nodes/reduce/reduceSum";
import { ReduceSumSquareNode } from "./nodes/reduce/reduceSumSquare";
import { ReluNode } from "./nodes/relu";
import { ReshapeNode } from "./nodes/reshape";
import { ShapeNode } from "./nodes/shape";
import { SliceNode } from "./nodes/slice";
import { SoftmaxNode } from "./nodes/softmax";
import { SubNode } from "./nodes/binary/sub";
import { TileNode } from "./nodes/tile";
import { TransposeNode } from "./nodes/transpose";
import { UnsqueezeNode } from "./nodes/unsqueeze";
import { UpsampleNode } from "./nodes/upsample";
import { NodeConstructor } from "./types";

export const nodeResolve: {[opType: string]: NodeConstructor} = {
  'Conv': (attributes, inputs, outputs, constants, onnxVersion) => new ConvNode(attributes, inputs, outputs, constants, onnxVersion),
  'BatchNormalization': (attributes, inputs, outputs, constants, onnxVersion) => new BatchNormalizationNode(attributes, inputs, outputs, constants, onnxVersion),
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
  'ReduceMax': (attributes, inputs, outputs, constants, onnxVersion) => new ReduceMaxNode(attributes, inputs, outputs, constants, onnxVersion),
  'ReduceSumSquare': (attributes, inputs, outputs, constants, onnxVersion) => new ReduceSumSquareNode(attributes, inputs, outputs, constants, onnxVersion),
  'Sub': (attributes, inputs, outputs, constants, onnxVersion) => new SubNode(attributes, inputs, outputs, constants, onnxVersion),
  'Mul': (attributes, inputs, outputs, constants, onnxVersion) => new MulNode(attributes, inputs, outputs, constants, onnxVersion),
  'Div': (attributes, inputs, outputs, constants, onnxVersion) => new DivNode(attributes, inputs, outputs, constants, onnxVersion),
  'Unsqueeze': (attributes, inputs, outputs, constants, onnxVersion) => new UnsqueezeNode(attributes, inputs, outputs, constants, onnxVersion),
  'Concat': (attributes, inputs, outputs, constants, onnxVersion) => new ConcatNode(attributes, inputs, outputs, constants, onnxVersion),
  'ConstantOfShape': (attributes, inputs, outputs, constants, onnxVersion) => new ConstantOfShapeNode(attributes, inputs, outputs, constants, onnxVersion),
  'Expand': (attributes, inputs, outputs, constants, onnxVersion) => new ExpandNode(attributes, inputs, outputs, constants, onnxVersion),
  'InstanceNormalization': (attributes, inputs, outputs, constants, onnxVersion) => new InstanceNormalizationNode(attributes, inputs, outputs, constants, onnxVersion),
  'Pad': (attributes, inputs, outputs, constants, onnxVersion) => new PadNode(attributes, inputs, outputs, constants, onnxVersion),
  'Relu': (attributes, inputs, outputs, constants, onnxVersion) => new ReluNode(attributes, inputs, outputs, constants, onnxVersion),
  'Shape': (attributes, inputs, outputs, constants, onnxVersion) => new ShapeNode(attributes, inputs, outputs, constants, onnxVersion),
  'Gather': (attributes, inputs, outputs, constants, onnxVersion) => new GatherNode(attributes, inputs, outputs, constants, onnxVersion),
  'Cast': (attributes, inputs, outputs, constants, onnxVersion) => new CastNode(attributes, inputs, outputs, constants, onnxVersion),
  'Floor': (attributes, inputs, outputs, constants, onnxVersion) => new FloorNode(attributes, inputs, outputs, constants, onnxVersion),
  'Ceil': (attributes, inputs, outputs, constants, onnxVersion) => new CeilNode(attributes, inputs, outputs, constants, onnxVersion),
  'Slice': (attributes, inputs, outputs, constants, onnxVersion) => new SliceNode(attributes, inputs, outputs, constants, onnxVersion),
  'Upsample': (attributes, inputs, outputs, constants, onnxVersion) => new UpsampleNode(attributes, inputs, outputs, constants, onnxVersion),
  'Transpose': (attributes, inputs, outputs, constants, onnxVersion) => new TransposeNode(attributes, inputs, outputs, constants, onnxVersion),
  'Softmax': (attributes, inputs, outputs, constants, onnxVersion) => new SoftmaxNode(attributes, inputs, outputs, constants, onnxVersion),
};