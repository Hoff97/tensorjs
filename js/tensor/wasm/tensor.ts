import Tensor from '../../types';

import { Tensor as WT } from 'rust-wasm-tensor';

let WASMT: typeof WT;
export let wasmLoaded: Promise<void> = new Promise((resolve, reject) => {
  import('rust-wasm-tensor').then(x => {
    WASMT = x.Tensor;
    resolve();
  });
});


export default class WASMTensor extends Tensor {
  public wasmTensor: WT;

  constructor(values: Float32Array | WT, shape?: Uint32Array) {
    super();

    if (values instanceof Float32Array) {
      if (shape === undefined) {
        throw new Error('Need the shape when creating a Wasm tensor from values');
      }
      this.wasmTensor = WASMT.create(shape, values);
    } else {
      this.wasmTensor = values;
    }
  }

  getValues() {
    return Promise.resolve(this.wasmTensor.get_vals());
  }

  getShape(): readonly number[] {
    return Array.from(this.wasmTensor.get_shape());
  }

  exp(): Tensor {
    return new WASMTensor(this.wasmTensor.exp());
  }

  log(): Tensor {
    return new WASMTensor(this.wasmTensor.log());
  }

  sqrt(): Tensor {
    return new WASMTensor(this.wasmTensor.sqrt());
  }

  add(tensor: Tensor): Tensor {
    if (!(tensor instanceof WASMTensor)) {
      throw new Error('Can only add WASM tensor to WASM tensor');
    }
    return new WASMTensor(this.wasmTensor.addition(tensor.wasmTensor));
  }

  subtract(tensor: Tensor): Tensor {
    if (!(tensor instanceof WASMTensor)) {
      throw new Error('Can only add WASM tensor to WASM tensor');
    }
    return new WASMTensor(this.wasmTensor.subtraction(tensor.wasmTensor));
  }

  multiply(tensor: Tensor): Tensor {
    if (!(tensor instanceof WASMTensor)) {
      throw new Error('Can only add WASM tensor to WASM tensor');
    }
    return new WASMTensor(this.wasmTensor.multiply(tensor.wasmTensor));
  }

  divide(tensor: Tensor): Tensor {
    if (!(tensor instanceof WASMTensor)) {
      throw new Error('Can only add WASM tensor to WASM tensor');
    }
    return new WASMTensor(this.wasmTensor.divide(tensor.wasmTensor));
  }

  matMul(tensor: Tensor): Tensor {
    if (!(tensor instanceof WASMTensor)) {
      throw new Error('Can only add WASM tensor to WASM tensor');
    }
    return new WASMTensor(this.wasmTensor.matmul(tensor.wasmTensor));
  }
}
