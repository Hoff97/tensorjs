import React from 'react';
import './App.css';
import { loadModel } from './inference';

import * as tjs from '@hoff97/tensor-js';

class App extends React.Component {
  private model?: tjs.onnx.model.OnnxModel = undefined;
  private modelCompiled = false;

  private scale = new tjs.tensor.gpu.GPUTensor(new Float32Array([255]), [1]);

  private videoTensor?: tjs.tensor.gpu.GPUTensor;

  constructor(props: {}) {
    super(props);

    loadModel('version-RFB-320').then(x => {
      if (x !== undefined) {
        this.model = x;

        this.getVideo();
      }
    });
  }

  async getVideo() {
    const video: HTMLVideoElement = (document.querySelector("#videoElement") as any);

    if (navigator.mediaDevices.getUserMedia) {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          height: 240,
          width: 320
        }
      });
      video.srcObject = stream;

      setTimeout(() => {
        this.videoTensor = tjs.tensor.gpu.GPUTensor.fromData(video);

        this.compileModel();
      }, 500);
    }
  }

  prepareVideo() {
    if (this.videoTensor !== undefined) {
      let [height, width] = this.videoTensor.shape.slice(0,2);

      const sliced = this.videoTensor.slice([0], [3], [2]);

      const transposed = sliced.transpose([2, 0, 1]);
      sliced.delete();
      const multiplied = transposed.multiply(this.scale)
      transposed.delete();

      const reshaped = multiplied.reshape([1,3,height,width], false);
      return reshaped;
    }
  }

  async compileModel() {
    if (this.videoTensor !== undefined && !this.modelCompiled) {
      this.modelCompiled = true;

      const reshaped = this.prepareVideo() as tjs.Tensor;

      await this.model?.compileForInputs([reshaped]);

      console.log('Compiled');
      console.log('Doing forward pass');

      this.model?.forward([reshaped]).then(result => {
        reshaped.delete();
        this.handleResult(result);
      });
    }
  }

  handleResult(tensors: tjs.Tensor[]) {
    console.log('Got result', tensors);

    requestAnimationFrame(() => {
      //@ts-ignore
      (tensors[0] as tjs.tensor.gpu.GPUTensor).delete(this.model.allocator);
      //@ts-ignore
      (tensors[1] as tjs.tensor.gpu.GPUTensor).delete(this.model.allocator);
      const reshaped = this.prepareVideo() as tjs.Tensor;

      this.model?.forward([reshaped]).then(results => {
        reshaped.delete();
        this.handleResult(results);
      });
    });
    /*this.setState({
      ...this.state,
      showResult: true
    })

    const sh = tensor.getShape();

    tensor = tensor.reshape(sh.slice(1), false);
    const transposed = tensor.transpose([1,2,0]);
    tensor.delete();

    const t = (transposed as tjs.tensor.gpu.GPUTensor).toTexture();
    transposed.delete();

    t.getValues().then(x => {
      const canv = document.getElementById("canvas") as HTMLCanvasElement;
      const context = canv.getContext("2d");

      if (context) {
        var id = context.createImageData(t.shape[0],t.shape[1]);
        var d  = id.data;

        for (let i = 0; i < x.length; i++) {
          d[i] = Math.round(x[i]);
        }
        context.putImageData(id, 0, 0);
      }

      t.delete();
    });*/
  }

  render() {
    return (
      <div className="App">
        <h1>Face detection</h1><br/>
        <video autoPlay id="videoElement"/>
      </div>
    );
  }
}

export default App;
