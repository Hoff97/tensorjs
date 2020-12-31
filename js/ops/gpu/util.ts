import { DrawCommand, Framebuffer2D } from "regl";
import GPUTensor, { gl } from "../../tensor/gpu/tensor";
import { getSize } from "../../util/shape";

export function buildComp(inputTextures: string[], fragmentShader: string) {
  const uniforms: any = {};
  for (let inputTexture of inputTextures) {
    uniforms[inputTexture] = gl.prop(inputTexture as never);
  }
  console.log(uniforms);

  const result = gl({
    frag: fragmentShader,
    vert: `
      precision mediump float;
      attribute vec2 position;
      varying vec2 uv;
      void main() {
        uv = 0.5 * (position + 1.0);
        gl_Position = vec4(position, 0, 1);
      }`,
    attributes: {
      position: [-4, -4, 4, -4, 0, 4]
    },
    uniforms: uniforms,
    depth: {
      enable: false
    },
    count: 3
  });
  return result;
}

export function compute(op: DrawCommand, resultShape: readonly number[], inputs: {[name: string]: GPUTensor}) {
  const inputTextures: {[name: string]: Framebuffer2D} = {};
  for (let name in inputs) {
    inputTextures[name] = inputs[name].framebuffer
  }

  console.log(inputTextures);

  const resultSize = getSize(resultShape);

  const result = gl.framebuffer({
    width: resultSize,
    height: 1,
    depthStencil: false,
    colorFormat: 'rgba',
    colorType: 'float'
  });

  op({
    framebuffer: result,
    ...inputTextures
  });

  return new GPUTensor(result, resultShape);
}