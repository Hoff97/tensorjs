import testBasic from './basic';
import WASMTensor from '../js/tensor/wasm/tensor';
import { wasmLoaded } from '../js/tensor/wasm/tensor';

testBasic('WASM', (shape: ReadonlyArray<number>, values: number[]) => {
  const sh = Uint32Array.from(shape);
  const vals = Float32Array.from(values);
  return new WASMTensor(vals, sh);
}, wasmLoaded);
