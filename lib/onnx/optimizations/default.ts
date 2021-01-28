import { ConvBatchNorm } from "./convBatchnorm";
import { ConvRelu } from "./convRelu";
import { Optimization } from "./optimization";

export const defaultOptimizations: Optimization[] = [
  new ConvBatchNorm(),
  new ConvRelu()
]