import REGL, {Framebuffer2D, Regl} from 'regl';
import {OrderedDict} from '../../util/datastructs/types';
import {primeFactors} from '../../util/math';
import {DTypeGpu} from './interface';

export interface MemoryEntry {
  width: number;
  height: number;
  size: number;

  frameBuffer: Framebuffer2D;
  dtype: DTypeGpu;

  id: number;
}

export interface Size {
  width: number;
  height: number;
}

export const colorType = {
  float32: 'float',
  float16: 'half float',
  int32: 'float',
  int16: 'float',
  int8: 'float',
  uint32: 'float',
  uint16: 'float',
  uint8: 'float',
};

const valsPerTexel = 4;

export class GPUMemoryAllocator {
  private trees: {[dtype: string]: OrderedDict<number, MemoryEntry>};

  private entryId: number;

  private regl: Regl;

  private maxSizeFactor: number;

  public totalAllocations = 0;

  constructor(
    regl: Regl,
    private orderedDictConstructor: () => OrderedDict<number, MemoryEntry>,
    maxSizeFactor?: number
  ) {
    this.trees = {};

    this.regl = regl;
    this.entryId = 0;

    this.maxSizeFactor = maxSizeFactor || 2;
  }

  getColorType(dtype: DTypeGpu) {
    //@ts-ignore
    return colorType[dtype as string] as string;
  }

  dtypeGroup(dtype: DTypeGpu) {
    if (dtype === 'float16') {
      return 'float16';
    } else {
      // We represent all other data types as float32 textures
      // This is of course technically not correct, but WebGL only
      // allows writing/reading float values from textures anyway
      // and the overhead of converting between the correct dtype and float32
      // in every shader is considered too big.
      return 'float32';
    }
  }

  allocate(size: number, dtype: DTypeGpu): MemoryEntry {
    const group = this.dtypeGroup(dtype);

    let upperBound = size * this.maxSizeFactor;
    const texSize = Math.ceil(size / valsPerTexel) * valsPerTexel;
    if (texSize < upperBound) {
      upperBound = texSize;
    }

    const results =
      this.trees[group] !== undefined
        ? this.trees[group].betweenBoundsFirst({
            gte: texSize,
            lte: upperBound,
          })
        : [];

    if (results.length === 0) {
      const textureSize = Math.ceil(size / valsPerTexel);
      const {width, height} = this.getTextureDims(textureSize);

      const framebuffer = this.regl.framebuffer({
        width: width,
        height: height,
        depthStencil: false,
        colorFormat: 'rgba',
        colorType: colorType[dtype] as any,
      });

      const memoryEntry: MemoryEntry = {
        width: width,
        height: height,
        size: width * height * valsPerTexel,
        frameBuffer: framebuffer,
        id: this.entryId++,
        dtype: dtype,
      };

      this.totalAllocations++;

      return memoryEntry;
    } else {
      const first = results[0];
      this.trees[group].deleteFirst(first.key);

      first.value.dtype = dtype;

      return first.value;
    }
  }

  getAllocationDimensions(size: number, dtype: DTypeGpu): Size {
    const group = this.dtypeGroup(dtype);

    let upperBound = size * this.maxSizeFactor;
    const texSize = Math.ceil(size / valsPerTexel) * valsPerTexel;
    if (texSize < upperBound) {
      upperBound = texSize;
    }

    const results =
      this.trees[group] !== undefined
        ? this.trees[group].betweenBoundsFirst({
            gte: size,
            lte: upperBound,
          })
        : [];

    if (results.length === 0) {
      const textureSize = Math.ceil(size / valsPerTexel);
      return this.getTextureDims(textureSize);
    } else {
      const first = results[0];

      return {
        width: first.value.width,
        height: first.value.height,
      };
    }
  }

  deallocate(entry: MemoryEntry) {
    const group = this.dtypeGroup(entry.dtype);

    if (this.trees[group] === undefined) {
      this.trees[group] = this.orderedDictConstructor();
    }
    this.trees[group].insert(entry.size, entry);
  }

  allocateTexture(values: number[], dtype: DTypeGpu): MemoryEntry {
    const textureSize = Math.ceil(values.length / valsPerTexel);
    const {width, height} = this.getTextureDims(textureSize);
    const arraySize = width * height * valsPerTexel;

    const vals = new Array(arraySize);
    for (let i = 0; i < values.length; i++) {
      vals[i] = values[i];
    }
    for (let i = values.length; i < arraySize; i++) {
      vals[i] = 0;
    }

    const texture = this.regl.texture({
      width: width,
      height: height,
      format: 'rgba',
      type: colorType[dtype] as any,
      // TODO: Convert data!
      data: vals,
    });

    const framebuffer = this.regl.framebuffer({
      color: texture,
      width: width,
      height: height,
      depthStencil: false,
    });

    this.totalAllocations++;

    return {
      width: width,
      height: height,
      size: arraySize,
      frameBuffer: framebuffer,
      id: this.entryId++,
      dtype: dtype,
    };
  }

  allocateOfDimensions(
    width: number,
    height: number,
    dtype: DTypeGpu
  ): MemoryEntry {
    const arraySize = width * height * valsPerTexel;

    const texture = this.regl.texture({
      width: width,
      height: height,
      format: 'rgba',
      type: colorType[dtype] as any,
    });

    const framebuffer = this.regl.framebuffer({
      color: texture,
      width: width,
      height: height,
      depthStencil: false,
    });

    this.totalAllocations++;

    return {
      width: width,
      height: height,
      size: arraySize,
      frameBuffer: framebuffer,
      id: this.entryId++,
      dtype: dtype,
    };
  }

  allocateFramebuffer(texture: REGL.Texture2D, dtype: DTypeGpu): MemoryEntry {
    const framebuffer = this.regl.framebuffer({
      color: texture,
      width: texture.width,
      height: texture.height,
      depthStencil: false,
    });

    this.totalAllocations++;

    return {
      width: texture.width,
      height: texture.height,
      size: texture.width * texture.height * valsPerTexel,
      frameBuffer: framebuffer,
      id: this.entryId++,
      dtype: dtype,
    };
  }

  private getTextureDims(size: number): Size {
    const factors = primeFactors(size);
    let width = 1;
    let height = 1;
    for (let i = 0; i < factors.length; i += 2) {
      width *= factors[i];
      if (i + 1 < factors.length) {
        height *= factors[i + 1];
      }
    }

    return {width, height};
  }

  public getNumEntries() {
    let num = 0;
    for (const dtype in this.trees) {
      num += this.trees[dtype].numEntries;
    }
    return num;
  }
}
