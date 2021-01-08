import { Framebuffer2D, Regl } from "regl";
import { AVLTree } from "../../util/avl";
import { primeFactors } from "../../util/math";

export interface MemoryEntry {
  width: number;
  height: number;
  size: number;

  frameBuffer: Framebuffer2D;

  id: number;
}

export class GPUMemoryAllocator {
  private tree: AVLTree<number,  MemoryEntry>;

  private entryId: number;

  private regl: Regl;

  private maxSizeFactor: number;

  constructor(regl: Regl, maxSizeFactor?: number) {
    this.tree = new AVLTree({
      compareValues: (a: MemoryEntry, b: MemoryEntry) => a.id === b.id
    });

    this.regl = regl;
    this.entryId = 0;

    this.maxSizeFactor = maxSizeFactor || 2;
  }

  allocate(size: number): MemoryEntry {
    let upperBound = size*this.maxSizeFactor;
    const texSize = Math.ceil(size/4)*4;
    if (texSize < upperBound) {
      upperBound = texSize;
    }

    const results = this.tree.betweenBoundsFirst({gte: size, lte: upperBound});
    if (results.length === 0) {
      const textureSize = Math.ceil(size / 4);
      const {width, height} = this.getTextureDims(textureSize);

      const framebuffer = this.regl.framebuffer({
        width: width,
        height: height,
        depthStencil: false,
        colorFormat: 'rgba',
        colorType: 'float'
      });

      const memoryEntry: MemoryEntry = {
        width: width,
        height: height,
        size: width*height*4,
        frameBuffer: framebuffer,
        id: this.entryId++
      };

      return memoryEntry;
    } else {
      const first = results[0];
      this.tree.deleteFirst(first.key);

      return first.value;
    }
  }

  deallocate(entry: MemoryEntry) {
    this.tree.insert(entry.size, entry);
  }

  allocateTexture(values: Float32Array): MemoryEntry {
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
      type: 'float',
      data: vals,
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
      id: this.entryId++
    }
  }

  private getTextureDims(size: number) {
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