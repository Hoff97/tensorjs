import React from 'react';
import './App.css';
import { loadModel } from './inference';

import * as tjs from '@hoff97/tensor-js';
import { GPUTensor } from '../../../dist/lib/tensor/gpu/tensor';
import { classes } from './classes';

interface AppState {
  img: any;
  scale: number;
  showResult: boolean;
}

class App extends React.Component<{}, AppState> {
  private model?: tjs.onnx.model.OnnxModel = undefined;

  private mean = new tjs.tensor.gpu.GPUTensor(new Float32Array([0.485, 0.456, 0.406]), [3,1,1], 16);
  private std = new tjs.tensor.gpu.GPUTensor(new Float32Array([0.229, 0.224, 0.225]), [3,1,1], 16);

  constructor(props: {}) {
    super(props);

    loadModel('mobilenet').then(x => {
      this.model = x;
      console.log('Got model');
    });

    this.setState({
      scale: 50
    });
  }

  async getImageData() {
    const el = document.getElementById("img") as HTMLImageElement;

    console.log('Reading pixels');
    const tensor = tjs.tensor.gpu.GPUTensor.fromData(el, 16);

    let [height, width] = tensor.shape.slice(0,2);

    const cropSize = Math.min(width, height);
    const halfSize = Math.floor(cropSize/2);

    const widthSliceStart = Math.floor(width/2) - halfSize;
    const heightSliceStart = Math.floor(height/2) - halfSize;

    const sliced = tensor.slice([heightSliceStart, widthSliceStart,0], [heightSliceStart + cropSize,widthSliceStart + cropSize,3], [0,1,2]);
    tensor.delete();

    const scale = 224/cropSize;
    const scaled = sliced.upsample([scale,scale,1]);
    sliced.delete();

    const croppedSize = scaled.getShape()[0];

    this.setState({
      ...this.state
    });

    const transposed = scaled.transpose([2, 0, 1]);
    scaled.delete();
    const shifted = transposed.subtract(this.mean);
    transposed.delete();
    const norm = shifted.divide(this.std);
    shifted.delete();
    const reshaped = norm.reshape([1,3,croppedSize,croppedSize], false);

    console.log('Doing forward pass');
    console.log(reshaped.getShape())
    this.model?.forward([reshaped]).then(result => this.handleResult(result[0]));
  }

  handleResult(tensor: tjs.Tensor) {
    console.log('Got result', tensor);

    this.setState({
      ...this.state,
      showResult: true
    });

    const probs = tensor.softmax(1);
    tensor.delete();

    (probs as tjs.tensor.gpu.GPUTensor).copy(32).getValues().then(x => {
      let probs = Array.from(x).map((v, i) => {
        return {
          prob: v,
          name: classes[i]
        }
      });

      probs = probs.sort((a, b) => b.prob-a.prob);
      console.log(probs);
    });
  }

  fileSelected(ev: React.ChangeEvent<HTMLInputElement>) {
    //@ts-ignore
    this.setState({
      ...this.state,
      scale: 50,
      //@ts-ignore
      img: URL.createObjectURL(ev.target.files[0]),
      showResult: false
    });
  }

  getImageWidth(scale: number) {
    const width = Math.round(400*(scale/50) + 50);

    return Math.floor(width/32)*32;
  }

  render() {
    let img;
    let scale = 50;
    let showResult = false;
    if (this.state) {
      img = this.state.img;
      scale = this.state.scale || 50;
      showResult = this.state.showResult;
    }

    const width = this.getImageWidth(scale);

    return (
      <div className="App">
        <h1>MobilenNet Classification</h1>
        <label htmlFor="file">Choose an image:</label> <input type='file' id="file" onChange={x => this.fileSelected(x)}/><br/>
        { img !== undefined ? (<>
            <img id="img" src={this.state.img} alt="Your upload" width={width}/><br/>

            <button onClick={() => this.getImageData()}>Run</button><br/>

            {
              showResult ? (
                "Result!"
              ) : (<></>)
            }
          </>) : (<></>)}
      </div>
    );
  }
}

export default App;
