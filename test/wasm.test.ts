import testBasic from './basic';
import testAggregate from './aggregate';
import WASMTensor from '../js/tensor/wasm/tensor';
import { wasmLoaded } from '../js/tensor/wasm/tensor';
import testConv from './conv';
import testPool from './pool';

const constructor = (shape: ReadonlyArray<number>, values: number[]) => {
  const sh = Uint32Array.from(shape);
  const vals = Float32Array.from(values);
  return new WASMTensor(vals, sh);
};

/*testBasic('WASM', constructor, wasmLoaded);
testPool('WASM', constructor, wasmLoaded);
testConv('WASM', constructor, wasmLoaded);*/

testPool('CPU', constructor, wasmLoaded);