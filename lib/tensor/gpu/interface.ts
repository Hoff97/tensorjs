import {Precision} from '../../types';
import {MemoryEntry} from './memory';

export interface GPUTensorI {
  memory: MemoryEntry;
  size: number;
  shape: readonly number[];
}

export type GPUTensorConstructor<GPUTensor extends GPUTensorI> = (
  memory: MemoryEntry,
  shape: readonly number[],
  precision: Precision
) => GPUTensor;
