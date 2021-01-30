import { GPUTensorI } from "../../../tensor/gpu/interface";
import { PoolOperation } from "./pool";

export class ProductOperation<GPUTensor extends GPUTensorI> extends PoolOperation<GPUTensor> {
  update(a: string, b: string): string {
    return `${a} * ${b}`;
  }
}