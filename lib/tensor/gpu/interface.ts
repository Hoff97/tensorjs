import {MemoryEntry} from './memory';

export type DTypeGpu =
  | 'float32'
  | 'float16'
  | 'int32'
  | 'int16'
  | 'int8'
  | 'uint32'
  | 'uint16'
  | 'uint8';

export interface GPUTensorI {
  memory: MemoryEntry;
  size: number;
  shape: readonly number[];
}

export type GPUTensorConstructor<GPUTensor extends GPUTensorI> = (
  memory: MemoryEntry,
  shape: readonly number[],
  dtype: DTypeGpu
) => GPUTensor;
