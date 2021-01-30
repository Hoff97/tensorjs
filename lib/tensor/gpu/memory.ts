import REGL, { Framebuffer2D, Regl } from "regl";
import { Precision } from "../../types";
import { AVLTree } from "../../util/avl";
import { primeFactors } from "../../util/math";

export interface MemoryEntry {
  width: number;
  height: number;
  size: number;

  frameBuffer: Framebuffer2D;
  precision: Precision;

  id: number;
}

export interface Size {
  width: number;
  height: number;
}

export class GPUMemoryAllocator {
  private trees: {[precision: number]: AVLTree<number,  MemoryEntry>};

  private entryId: number;

  private regl: Regl;

  private maxSizeFactor: number;

  constructor(regl: Regl, maxSizeFactor?: number) {
    this.trees = {
      16: new AVLTree({
        compareValues: (a: MemoryEntry, b: MemoryEntry) => a.id === b.id
      }),
      32: new AVLTree({
        compareValues: (a: MemoryEntry, b: MemoryEntry) => a.id === b.id
      })
    };

    this.regl = regl;
    this.entryId = 0;

    this.maxSizeFactor = maxSizeFactor || 2;
  }

  allocate(size: number, precision: Precision): MemoryEntry {
    let upperBound = size*this.maxSizeFactor;
    const texSize = Math.ceil(size/4)*4;
    if (texSize < upperBound) {
      upperBound = texSize;
    }

    const results = this.trees[precision].betweenBoundsFirst({gte: size, lte: upperBound});
    if (results.length === 0) {
      const textureSize = Math.ceil(size / 4);
      const {width, height} = this.getTextureDims(textureSize);

      const framebuffer = this.regl.framebuffer({
        width: width,
        height: height,
        depthStencil: false,
        colorFormat: 'rgba',
        colorType: precision === 32 ? 'float' : 'half float'
      });

      const memoryEntry: MemoryEntry = {
        width: width,
        height: height,
        size: width*height*4,
        frameBuffer: framebuffer,
        id: this.entryId++,
        precision: precision
      };

      return memoryEntry;
    } else {
      const first = results[0];
      this.trees[precision].deleteFirst(first.key);

      return first.value;
    }
  }

  getAllocationDimensions(size: number, precision: Precision): Size {
    let upperBound = size*this.maxSizeFactor;
    const texSize = Math.ceil(size/4)*4;
    if (texSize < upperBound) {
      upperBound = texSize;
    }

    const results = this.trees[precision].betweenBoundsFirst({gte: size, lte: upperBound});
    if (results.length === 0) {
      const textureSize = Math.ceil(size / 4);
      return this.getTextureDims(textureSize);
    } else {
      const first = results[0];

      return {
        width: first.value.width,
        height: first.value.height
      }
    }
  }

  deallocate(entry: MemoryEntry) {
    this.trees[entry.precision].insert(entry.size, entry);
  }

  allocateTexture(values: Float32Array, precision: Precision): MemoryEntry {
    const textureSize = Math.ceil(values.length/4);
    const {width, height} = this.getTextureDims(textureSize);
    const arraySize = width*height*4;

    const vals = new Float32Array(arraySize);
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
      type: precision === 32 ? 'float' : 'half float',
      data: precision === 32 ? vals : Array.from(vals),
    });

    const framebuffer = this.regl.framebuffer({
      color: texture,
      width: width,
      height: height,
      depthStencil: false
    });

    return {
      width: width,
      height: height,
      size: arraySize,
      frameBuffer: framebuffer,
      id: this.entryId++,
      precision: precision
    }
  }

  allocateOfDimensions(width: number, height: number, precision: Precision): MemoryEntry {
    const arraySize = width*height*4;

    const texture = this.regl.texture({
      width: width,
      height: height,
      format: 'rgba',
      type: precision === 32 ? 'float' : 'half float',
    });

    const framebuffer = this.regl.framebuffer({
      color: texture,
      width: width,
      height: height,
      depthStencil: false
    });

    return {
      width: width,
      height: height,
      size: arraySize,
      frameBuffer: framebuffer,
      id: this.entryId++,
      precision: precision
    }
  }

  allocateFramebuffer(texture: REGL.Texture2D, precision: Precision): MemoryEntry {
    const framebuffer = this.regl.framebuffer({
      color: texture,
      width: texture.width,
      height: texture.height,
      depthStencil: false
    });

    return {
      width: texture.width,
      height: texture.height,
      size: texture.width*texture.height*4,
      frameBuffer: framebuffer,
      id: this.entryId++,
      precision: precision
    }
  }

  private getTextureDims(size: number): Size {
    const factors = primeFactors(size);
    let width = 1;
    let height = 1;
    for (let i = 0; i < factors.length; i+=2) {
      width *= factors[i];
      if (i + 1 < factors.length) {
        height *= factors[i+1];
      }
    }

    return {width, height};
  }
}