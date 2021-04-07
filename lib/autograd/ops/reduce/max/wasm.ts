import {DTypeWasm, WASMTensor} from '../../../../tensor/wasm/tensor';

export function maxBackWASM<DTpe extends DTypeWasm>(
  value: WASMTensor<DTpe>,
  gradient: WASMTensor<DTpe>,
  axes: number[]
) {
  return new WASMTensor(
    gradient.wasmTensor.max_backward(
      value.wasmTensor as any,
      new Uint32Array(axes)
    ) as any,
    undefined,
    value.dtype
  );
}
