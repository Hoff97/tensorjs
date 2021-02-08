import REGL from 'regl';
import {Dict} from '../../util/datastructs/dict';
import {GPUMemoryAllocator} from './memory';

const canvas = document.createElement('canvas');

export let glContext: WebGLRenderingContext;
export let gl: REGL.Regl;
export let defaultAllocator: GPUMemoryAllocator;

function setup() {
  //@ts-ignore
  glContext = canvas.getContext('webgl', {
    failIfMajorPerformanceCaveat: false,
  });

  gl = REGL({
    gl: glContext,
    extensions: [
      'OES_texture_float',
      'WEBGL_color_buffer_float',
      'OES_texture_half_float',
    ],
  });

  defaultAllocator = new GPUMemoryAllocator(gl, () => {
    return new Dict((key: number) => key);
  });
}

setup();
