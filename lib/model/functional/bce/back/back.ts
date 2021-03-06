import {BackwardOp, VariableI} from '../../../../autograd/types';
import {Tensor} from '../../../../library';
import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {GPUTensor} from '../../../../tensor/gpu/tensor';
import {WASMTensor} from '../../../../tensor/wasm/tensor';
import {DType} from '../../../../types';
import {bceBack} from './cpu';
import {defaultBCEBackD} from './gpu';

export class BCEBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public x: VariableI<DTpe>, public y: VariableI<DTpe>) {}

  backward(grad: Tensor<DTpe>): void {
    let gradX: Tensor<DTpe>;
    if (grad instanceof CPUTensor) {
      const back = bceBack(
        this.x.value as CPUTensor<DTpe>,
        this.y.value as CPUTensor<DTpe>
      );
      gradX = grad.multiply(back);
      back.delete();
    } else if (grad instanceof WASMTensor) {
      const back = ((this.x.value as WASMTensor<any>)
        .wasmTensor as any).bce_back(
        (this.y.value as WASMTensor<any>).wasmTensor
      );
      gradX = new WASMTensor(
        grad.wasmTensor.multiply(back, 1.0),
        grad.dtype
      ) as WASMTensor<any>;
      back.free();
    } else {
      const back = defaultBCEBackD.calc(
        {
          A: this.x.value as GPUTensor<any>,
          B: this.y.value as GPUTensor<any>,
          outputShape: this.x.getShape(),
        },
        this.x.value.dtype as any
      ) as GPUTensor<any>;
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
