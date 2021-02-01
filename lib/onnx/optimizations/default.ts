import {ConvBatchNorm} from './convBatchnorm';
import {ConvRelu} from './convRelu';
import {ConvRelu6} from './convRelu6';
import {Optimization} from './optimization';

export const defaultOptimizations: Optimization[] = [
  new ConvBatchNorm(),
  new ConvRelu(),
  new ConvRelu6(),
];
