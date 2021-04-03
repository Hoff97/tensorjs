import {WASMTensor} from '../../../tensor/wasm/tensor';

export function addIndexWASM(
  indices: WASMTensor<'uint32'>,
  axis: number,
  count: number
): WASMTensor<'uint32'> {
  return new WASMTensor(indices.wasmTensor.add_index(axis, count));
}
