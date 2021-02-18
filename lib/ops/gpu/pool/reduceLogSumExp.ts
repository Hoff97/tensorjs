import {GPUTensorI} from '../../../tensor/gpu/interface';
import {PoolOperation} from './pool';

export class ReduceLogSumExpOperation<
  GPUTensor extends GPUTensorI
> extends PoolOperation<GPUTensor> {
  update(a: string, b: string): string {
    return `exp(${a}) + ${b}`;
  }
  post(res: string) {
    return `${res} = log(${res});`;
  }

  init(res: string) {
    return `exp(${res})`;
  }
}
