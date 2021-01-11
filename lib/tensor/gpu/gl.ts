import REGL from 'regl';
import { GPUMemoryAllocator } from './memory';

const canvas = document.createElement("canvas");

export let glContext: WebGLRenderingContext;
export let gl: REGL.Regl;
export let defaultAllocator: GPUMemoryAllocator;

function setup() {
  glContext = canvas.getContext("webgl", {
    failIfMajorPerformanceCaveat: false
  });

  gl = REGL({
    gl: glContext,
    extensions: ['OES_texture_float', 'WEBGL_color_buffer_float']
  });

  defaultAllocator = new GPUMemoryAllocator(gl);
}

setup();