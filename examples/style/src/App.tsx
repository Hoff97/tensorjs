import React from 'react';
import logo from './logo.svg';
import './App.css';
import { loadModel } from './inference';

import * as tjs from '@hoff97/tensor-js';

interface AppState {
  img: any;
  scale: number;
  croppedSize: number;
  showResult: boolean;
  model: string;
}

const models = [
  "mosaic",
  "candy",
  "pointilism",
  "udnie",
  "rain-princess"
];

class App extends React.Component<{}, AppState> {
  private model?: tjs.onnx.model.OnnxModel = undefined;

  private scale = new tjs.tensor.gpu.GPUTensor(new Float32Array([255]), [1]);

  constructor(props: {}) {
    super(props);

    loadModel('mosaic').then(x => {
      this.model = x;
    });

    this.setState({
      scale: 50,
      model: 'mosaic'
    });
  }

  getImageData() {
    const el = document.getElementById("img") as HTMLImageElement;

    console.log('Reading pixels');
    const tensor = tjs.tensor.gpu.GPUTensor.fromData(el);

    let [height, width] = tensor.shape.slice(0,2);

    const cropSize = Math.min(width, height);
    const halfSize = Math.floor(cropSize/2);

    const widthSliceStart = Math.floor(width/2) - halfSize;
    const heightSliceStart = Math.floor(height/2) - halfSize;

    const sliced = tensor.slice([heightSliceStart, widthSliceStart,0], [heightSliceStart + cropSize,widthSliceStart + cropSize,3], [0,1,2]);
    tensor.delete();

    const showWidth = this.getImageWidth(this.state.scale);
    const scale = showWidth/width;
    const scaled = sliced.upsample([scale,scale,1]);
    sliced.delete();

    const croppedSize = scaled.getShape()[0];

    this.setState({
      ...this.state,
      croppedSize: croppedSize,
    });

    const transposed = scaled.transpose([2, 0, 1]);
    scaled.delete();
    const multiplied = transposed.multiply(this.scale)
    transposed.delete();
    const reshaped = multiplied.reshape([1,3,croppedSize,croppedSize], false);

    console.log('Doing forward pass');
    this.model?.forward([reshaped], 100).then(result => this.handleResult(result[0]));
  }

  handleResult(tensor: tjs.Tensor) {
    console.log('Got result', tensor);

    this.setState({
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

  async setModel(name: string) {
    this.setState({
      ...this.state,
      model: name
    })
    this.model = await loadModel(name);
  }

  render() {
    let img;
    let scale = 50;
    let croppedSize = 50;
    let showResult = false;
    if (this.state) {
      img = this.state.img;
      scale = this.state.scale || 50;
      croppedSize = this.state.croppedSize;
      showResult = this.state.showResult;
    }

    const width = this.getImageWidth(scale);

    return (
      <div className="App">
        <h1>Style transfer</h1>
        <label htmlFor="model">Choose a style:</label> <select id="model" onChange={x => this.setModel(x.target.value)}>
          {
            models.map(x => (
              <option value={x} key={x}>{x}</option>
            ))
          }
        </select><br/>
        <label htmlFor="file">Choose an image:</label> <input type='file' id="file" onChange={x => this.fileSelected(x)}/><br/>
        { img !== undefined ? (<>
            <div className="slidecontainer">
              Scale: <input type="range" min="1" max="100" defaultValue={scale}
                className="slider" id="myRange"
                //@ts-ignore
                onChange={ev => this.setState({...this.state, showResult: false, scale: parseInt(ev.currentTarget.value)})}/>
            </div>
            <img id="img" src={this.state.img} alt="your image" width={width}/><br/>

            <button onClick={() => this.getImageData()}>Run</button><br/>

            {
              showResult ? (
                <canvas id="canvas" width={croppedSize} height={croppedSize}></canvas>
              ) : (<></>)
            }
          </>) : (<></>)}
      </div>
    );
  }
}

export default App;
