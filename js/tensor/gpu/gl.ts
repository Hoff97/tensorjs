import REGL from 'regl';
import { GPUMemoryAllocator } from './memory';

const canvas = document.createElement("canvas");

let glContext: WebGLRenderingContext;
export let gl: REGL.Regl;
export let defaultAllocator: GPUMemoryAllocator;

function setup() {
  glContext = canvas.getContext("webgl", {
    preserveDrawingBuffer: true,
    failIfMajorPerformanceCaveat: true
  });

  gl = REGL({
    gl: glContext,
    extensions: ['OES_texture_float']
  });

  defaultAllocator = new GPUMemoryAllocator(gl);
}

setup();