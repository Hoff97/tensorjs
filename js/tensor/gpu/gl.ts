import REGL from 'regl';
import { GPUMemoryAllocator } from './memory';

let glContext = document.createElement("canvas").getContext("webgl", {
  preserveDrawingBuffer: true,
  failIfMajorPerformanceCaveat: true
});

export let gl = REGL({
  gl: glContext,
  extensions: ['OES_texture_float']
});

export const defaultAllocator = new GPUMemoryAllocator(gl);