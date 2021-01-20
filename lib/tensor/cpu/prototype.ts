import { MemoryEntry } from '../gpu/memory';
import { CPUTensor } from './tensor';

export class PrototypeTensor extends CPUTensor {
  public memory: MemoryEntry;

  constructor(shape: ReadonlyArray<number>, memory: MemoryEntry, type?: string) {
    super(shape, null, type);

    this.memory = memory;
  }
}
