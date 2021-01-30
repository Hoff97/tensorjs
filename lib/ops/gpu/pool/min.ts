import { GPUTensorI } from "../../../tensor/gpu/interface";
import { PoolInput, PoolOperation } from "./pool";

export class MinOperation<GPUTensor extends GPUTensorI> extends PoolOperation<GPUTensor> {
  update(a: string, b: string): string {
    return `min(${a}, ${b})`;
  }
}
