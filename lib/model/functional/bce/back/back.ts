import {BackwardOp, VariableI} from '../../../../autograd/types';
import {Tensor} from '../../../../library';
import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {GPUTensor} from '../../../../tensor/gpu/tensor';
import {WASMTensor} from '../../../../tensor/wasm/tensor';
import {bceBack} from './cpu';
import {defaultBCEBackD} from './gpu';

export class BCEBack implements BackwardOp {
  constructor(public x: VariableI, public y: VariableI) {}

  backward(grad: Tensor): void {
    let gradX: Tensor;
    if (grad instanceof CPUTensor) {
      const back = bceBack(
        this.x.value as CPUTensor,
        this.y.value as CPUTensor
      );
      gradX = grad.multiply(back);
      back.delete();
    } else if (grad instanceof WASMTensor) {
      const back = (this.x.value as WASMTensor).wasmTensor.bce_back(
        (this.y.value as WASMTensor).wasmTensor
      );
      gradX = new WASMTensor(grad.wasmTensor.multiply(back, 1.0));
      back.free();
    } else {
      const back = defaultBCEBackD.calc(
        {
          A: this.x.value as GPUTensor,
          B: this.y.value as GPUTensor,
          outputShape: this.x.getShape(),
        },
        (this.x.value as GPUTensor).precision
      ) as GPUTensor;
      gradX = grad.multiply(back);
      back.delete();
    }
    const needed = this.x.backward(gradX);
    if (!needed) {
      gradX.delete();
    }
  }

  delete(): void {
    if (!this.x.isLeaf()) {
      this.x.delete();
    }

    if (!this.y.isLeaf()) {
      this.y.delete();
    }
  }
}
