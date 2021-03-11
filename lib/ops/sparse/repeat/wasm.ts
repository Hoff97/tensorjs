import {WASMTensor} from '../../../tensor/wasm/tensor';

export function repeatIndicesWASM(
  indices: WASMTensor<'uint32'>,
  repeats: readonly number[],
  shape: readonly number[],
  repeatsProd: number
): WASMTensor<'uint32'> {
  return new WASMTensor(
    indices.wasmTensor.repeat_sparse_indices(
      new Uint32Array(repeats),
      new Uint32Array(shape),
      repeatsProd
    )
  );
}
