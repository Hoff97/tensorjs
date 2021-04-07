import {DTypeWasm, WASMTensor} from '../../../../tensor/wasm/tensor';

export function minBackWASM<DTpe extends DTypeWasm>(
  value: WASMTensor<DTpe>,
  gradient: WASMTensor<DTpe>,
  axes: number[]
) {
  return new WASMTensor(
    gradient.wasmTensor.min_backward(
      value.wasmTensor as any,
      new Uint32Array(axes)
    ) as any,
    undefined,
    value.dtype
  );
}
