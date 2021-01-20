import { GPUTensorI } from "../../tensor/gpu/interface";
import { PoolInput, PoolOperation } from "./pool";

export class SumSquareOperation<GPUTensor extends GPUTensorI> extends PoolOperation<GPUTensor> {
  update(a: string, b: string): string {
    return `(${a}*${a}) + ${b}`;
  }
  init(res: string) {
    return `${res}*${res}`;
  }
}
