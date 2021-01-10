import {onnx} from 'onnx-proto';

import Tensor from '../types';
import { CPUTensor } from "../tensor/cpu/tensor";
import { TENSOR_FLOAT, TENSOR_INT64 } from './definitions';
import Long from 'long';

export function createTensor(tensorProto: onnx.ITensorProto): Tensor {
  if (tensorProto.segment !== undefined && tensorProto.segment !== null) {
    throw new Error('Handling of tensor proto segment not yet implemented');
  }

  let shape: number[] = tensorProto.dims as number[];
  if (shape === undefined || shape === null) {
    throw new Error('Tensor shape must be specified');
  }
  for (let i = 0; i < shape.length; i++) {
    if (Long.isLong(shape[i])) {
      shape[i] = (shape[i] as any).toNumber();
    }
  }
  if (shape.length === 0) {
    shape = [1];
  }

  if (tensorProto.dataType === TENSOR_FLOAT) {
    if (tensorProto.floatData && tensorProto.floatData.length > 0) {
      return new CPUTensor(shape, tensorProto.floatData);
    } else if (tensorProto.rawData && tensorProto.rawData.length > 0) {
      const buffer = tensorProto.rawData.buffer.slice(tensorProto.rawData.byteOffset, tensorProto.rawData.byteOffset+tensorProto.rawData.byteLength);
      const values = new Float32Array(buffer);
      return new CPUTensor(shape, values);
    } else {
      throw new Error('Cant process float tensor without float or raw data');
    }
  } else if (tensorProto.dataType === TENSOR_INT64) {
    if (tensorProto.rawData && tensorProto.rawData.length > 0) {
      const values = [];
      for (let i = 0; i < tensorProto.rawData.length; i += 8) {
        const value = Long.fromBytesLE(Array.from(tensorProto.rawData.slice(i,i+8))).toNumber();
        values.push(value);
      }

      return new CPUTensor(shape, values, "int");
    } else {
      throw new Error('Cant process int64 tensor without raw data');
    }
  } else {
    throw new Error(`Handling of tensor type ${tensorProto.dataType} not yet implemented`);
  }
}