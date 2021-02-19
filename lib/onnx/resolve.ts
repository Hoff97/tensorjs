import {AddNode} from './nodes/binary/add';
import {BatchNormalizationNode} from './nodes/batchNormalization';
import {CastNode} from './nodes/cast';
import {CeilNode} from './nodes/unary/ceil';
import {ClipNode} from './nodes/clip';
import {ConcatNode} from './nodes/nary/concat';
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
import {LogNode} from './nodes/unary/log';
import {SqrtNode} from './nodes/unary/sqrt';
import {SignNode} from './nodes/unary/sign';
import {ATanHNode, ATanNode, TanHNode, TanNode} from './nodes/unary/tan';
import {ACosHNode, ACosNode, CosHNode, CosNode} from './nodes/unary/cos';
import {ASinHNode, ASinNode, SinHNode, SinNode} from './nodes/unary/sin';
import {SigmoidNode} from './nodes/unary/sigmoid';
import {ReduceMinNode} from './nodes/reduce/reduceMin';
import {ReduceProdNode} from './nodes/reduce/reduceProd';
import {ReduceLogSumNode} from './nodes/reduce/reduceLogSum';
import {ReduceLogSumExpNode} from './nodes/reduce/reduceLogSumExp';
import {ReduceL2Node} from './nodes/reduce/reduceL2';
import {ReduceL1Node} from './nodes/reduce/reduceL1';
import {PowNode} from './nodes/binary/pow';
import {IdentityNode} from './nodes/unary/identity';
import {HardSigmoidNode} from './nodes/unary/hardSigmoid';
import {NegNode} from './nodes/unary/neg';
import {ReciprocalNode} from './nodes/unary/reciprocal';
import {SqueezeNode} from './nodes/squeeze';
import {SizeNode} from './nodes/size';
import {LeakyReluNode} from './nodes/leakyRelu';
import {EluNode} from './nodes/elu';
import {PReluNode} from './nodes/prelu';
import {SeluNode} from './nodes/selu';
import {FlattenNode} from './nodes/flatten';
import {SoftplusNode} from './nodes/softplus';
import {SoftsignNode} from './nodes/softsign';
import {SumNode} from './nodes/nary/sum';
import {MeanNode} from './nodes/nary/mean';
import {CeluNode} from './nodes/celu';
import {RoundNode} from './nodes/unary/round';
import {RangeNode} from './nodes/range';

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
  Pow: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new PowNode(attributes, inputs, outputs, constants, onnxVersion, mode),
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
  Range: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new RangeNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Flatten: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new FlattenNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Tile: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new TileNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  MatMul: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new MatMulNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Exp: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ExpNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Log: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new LogNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Sqrt: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new SqrtNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Sign: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new SignNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  ReduceSum: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ReduceSumNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
  ReduceProd: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ReduceProdNode(
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
  ReduceMin: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ReduceMinNode(
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
  ReduceL2: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ReduceL2Node(attributes, inputs, outputs, constants, onnxVersion, mode),
  ReduceL1: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ReduceL1Node(attributes, inputs, outputs, constants, onnxVersion, mode),
  ReduceLogSum: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ReduceLogSumNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
  ReduceLogSumExp: (
    attributes,
    inputs,
    outputs,
    constants,
    onnxVersion,
    mode
  ) =>
    new ReduceLogSumExpNode(
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
  Sum: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new SumNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Mean: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new MeanNode(attributes, inputs, outputs, constants, onnxVersion, mode),
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
  Squeeze: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new SqueezeNode(attributes, inputs, outputs, constants, onnxVersion, mode),
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
  LeakyRelu: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new LeakyReluNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
  Elu: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new EluNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  PRelu: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new PReluNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Selu: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new SeluNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Celu: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new CeluNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Shape: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ShapeNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Size: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new SizeNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Gather: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new GatherNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Cast: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new CastNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Floor: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new FloorNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Ceil: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new CeilNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Round: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new RoundNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Abs: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new AbsNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Neg: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new NegNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Reciprocal: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ReciprocalNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
  Identity: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new IdentityNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Sigmoid: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new SigmoidNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  HardSigmoid: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new HardSigmoidNode(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      mode
    ),
  Sin: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new SinNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Cos: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new CosNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Tan: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new TanNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Asin: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ASinNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Acos: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ACosNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Atan: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ATanNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Sinh: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new SinHNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Cosh: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new CosHNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Tanh: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new TanHNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Asinh: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ASinHNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Acosh: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ACosHNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Atanh: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new ATanHNode(attributes, inputs, outputs, constants, onnxVersion, mode),
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
  Softplus: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new SoftplusNode(attributes, inputs, outputs, constants, onnxVersion, mode),
  Softsign: (attributes, inputs, outputs, constants, onnxVersion, mode) =>
    new SoftsignNode(attributes, inputs, outputs, constants, onnxVersion, mode),
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
