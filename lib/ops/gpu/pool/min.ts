import { GPUTensorI } from "../../../tensor/gpu/interface";
import { PoolOperation } from "./pool";

export class MinOperation<GPUTensor extends GPUTensorI> extends PoolOperation<GPUTensor> {
  update(a: string, b: string): string {
    return `min(${a}, ${b})`;
  }
}
