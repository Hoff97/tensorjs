import testBasic from './basic';
import testAggregate from './aggregate';
import WASMTensor from '../lib/tensor/wasm/tensor';
import { wasmLoaded } from '../lib/tensor/wasm/tensor';
import testConv from './conv';
import testPool from './pool';

const constructor = (shape: ReadonlyArray<number>, values: number[]) => {
  const sh = Uint32Array.from(shape);
  const vals = Float32Array.from(values);
  return new WASMTensor(vals, sh);
};

testBasic('WASM', constructor, wasmLoaded);
testAggregate('WASM', constructor, wasmLoaded);
testConv('WASM', constructor, wasmLoaded);
testPool('CPU', constructor, wasmLoaded);