import { ConvBatchNorm } from "./convBatchnorm";
import { Optimization } from "./optimization";

export const defaultOptimizations: Optimization[] = [
  new ConvBatchNorm()
]