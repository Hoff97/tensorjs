import {AddNode} from './nodes/binary/add';
import {BatchNormalizationNode} from './nodes/batchNormalization';
import {CastNode} from './nodes/cast';
import {CeilNode} from './nodes/unary/ceil';
import {ClipNode} from './nodes/clip';
import {ConcatNode} from './nodes/concat';
import {ConstantNode} from './nodes/constant';
import {ConstantOfShapeNode} from './nodes/constantOfShape';
import {ConvNode} from './nodes/conv/conv';
import {DivNode} from './nodes/binary/div';
import {ExpNode} from './nodes/unary/exp';
import {ExpandNode} from './nodes/expand';
import {FloorNode} from './nodes/unary/floor';
import {GatherNode} from './nodes/gather';
import {GemmNode} from './nodes/gemm';
import {InstanceNormalizationNode} from './nodes/conv/instanceNormalization';
import {MatMulNode} from './nodes/matMul';
import {MulNode} from './nodes/binary/mul';
import {PadNode} from './nodes/conv/pad';
import {ReduceMaxNode} from './nodes/reduce/reduceMax';
import {ReduceMeanNode} from './nodes/reduce/reduceMean';
import {ReduceSumNode} from './nodes/reduce/reduceSum';
import {ReduceSumSquareNode} from './nodes/reduce/reduceSumSquare';
import {ReluNode} from './nodes/relu';
import {ReshapeNode} from './nodes/reshape';
import {ShapeNode} from './nodes/shape';
import {SliceNode} from './nodes/slice';
import {SoftmaxNode} from './nodes/softmax';
import {SubNode} from './nodes/binary/sub';
import {TileNode} from './nodes/tile';
import {TransposeNode} from './nodes/transpose';
import {UnsqueezeNode} from './nodes/unsqueeze';
import {UpsampleNode} from './nodes/upsample';
import {NodeConstructor} from './types';
import {GlobalAveragePoolNode} from './nodes/conv/globalAveragePool';
import {AbsNode} from './nodes/unary/abs';

export const nodeResolve: {[opType: string]: NodeConstructor} = {
  Conv: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ConvNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  BatchNormalization: (
    attributes,
    inputs,
    outputs,
    constants,
    onnxVersion,
    mode
  ) =>
    new BatchNormalizationNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
  Clip: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ClipNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Add: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new AddNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  ReduceMean: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ReduceMeanNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
  Gemm: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new GemmNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Constant: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ConstantNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Reshape: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ReshapeNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Tile: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new TileNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  MatMul: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new MatMulNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Exp: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ExpNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  ReduceSum: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ReduceSumNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
  ReduceMax: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ReduceMaxNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
  ReduceSumSquare: (
    attributes,
    inputs,
    outputs,
    constants,
    onnxVersion,
    mode
  ) =>
    new ReduceSumSquareNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
  Sub: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new SubNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Mul: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new MulNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Div: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new DivNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Unsqueeze: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new UnsqueezeNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
  Concat: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ConcatNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  ConstantOfShape: (
    attributes,
    inputs,
    outputs,
    constants,
    onnxVersion,
    mode
  ) =>
    new ConstantOfShapeNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
  Expand: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ExpandNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  InstanceNormalization: (
    attributes,
    inputs,
    outputs,
    constants,
    onnxVersion,
    mode
  ) =>
    new InstanceNormalizationNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
  Pad: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new PadNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Relu: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ReluNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Shape: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ShapeNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Gather: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new GatherNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Cast: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new CastNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Floor: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new FloorNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Ceil: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new CeilNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Abs: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new AbsNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Slice: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new SliceNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Upsample: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new UpsampleNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Transpose: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new TransposeNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
  Softmax: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new SoftmaxNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  GlobalAveragePool: (
    attributes,
    inputs,
    outputs,
    constants,
    onnxVersion,
    mode
  ) =>
    new GlobalAveragePoolNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
};
