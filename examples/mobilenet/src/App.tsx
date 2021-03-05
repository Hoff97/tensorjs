import React from 'react';
import './App.css';
import { loadModel } from './inference';

import * as tjs from '@hoff97/tensor-js';
import { classes } from './classes';

interface Result {
  name: string;
  prob: number;
}

interface AppState {
  img: any;
  showResult: boolean;
  results: Result[];
}

const imgs = [
  "n01440764_tench.JPEG",
  "n03160309_dam.JPEG",
  "n03216828_dock.JPEG",
  "n03670208_limousine.JPEG",
  "n04548280_wall_clock.JPEG",
  "n04552348_warplane.JPEG",
  "n04599235_wool.JPEG",
  "n06874185_traffic_light.JPEG",
  "n07873807_pizza.JPEG",
  "n07920052_espresso.JPEG",
  "n09193705_alp.JPEG",
  "n12620546_hip.JPEG",
];

class App extends React.Component<{}, AppState> {
  private model?: tjs.onnx.model.OnnxModel = undefined;

  private mean = new tjs.tensor.gpu.GPUTensor([0.485, 0.456, 0.406], [3,1,1]);
  private std = new tjs.tensor.gpu.GPUTensor([0.229, 0.224, 0.225], [3,1,1]);

  constructor(props: {}) {
    super(props);

    loadModel('mobilenet').then(x => {
      this.model = x;
      console.log('Got model');
    });

    this.setState({
      showResult: false,
      results: []
    });
  }

  async getImageData() {
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

    this.model?.forward([reshaped]).then(result => this.handleResult(result[0]));
  }

  async handleResult(tensor: tjs.Tensor) {
    console.log('Got result', tensor);

    const probs = tensor.softmax(1);
    tensor.delete();

    const values = await (probs as tjs.tensor.gpu.GPUTensor).getValues()
    let probMap = Array.from(values).map((v, i) => {
      return {
        prob: v,
        name: classes[i]
      }
    });

    probMap = probMap.sort((a, b) => b.prob-a.prob);

    this.setState({
      ...this.state,
      showResult: true,
      results: probMap.slice(0, 5)
    });
  }

  fileSelected(ev: React.ChangeEvent<HTMLInputElement>) {
    //@ts-ignore
    this.setState({
      ...this.state,
      //@ts-ignore
      img: URL.createObjectURL(ev.target.files[0]),
      showResult: false,
      results: []
    });
  }

  setImage(img: string) {
    this.setState({
      ...this.state,
      //@ts-ignore
      img: "img/" + img,
      showResult: false,
      results: []
    });
  }

  render() {
    let img;
    let showResult = false;
    if (this.state) {
      img = this.state.img;
      showResult = this.state.showResult;
    }

    const width = 400;

    return (
      <div className="App">
        <h1>MobilenNet Classification</h1>
        <label htmlFor="file">Choose an image:</label> <input type='file' id="file" onChange={x => this.fileSelected(x)}/><br/>
        Or use one of the examples:
        <table>
          <tr>
            {imgs.map(img => (
              <td>
                <img src={"img/" + img}
                  height={50} onClick={() => this.setImage(img)}
                  className="exampleImage" alt="Example"></img>
              </td>
            ))}
          </tr>
        </table>
        { img !== undefined ? (<>
          <table>
            <tr>
              <td>
                <img id="img" src={this.state.img} alt="Your upload" width={width}/><br/>

                <button onClick={() => this.getImageData()}>Run</button><br/>
              </td>
              <td className="resultList">
                {
                  showResult ? (
                    this.renderResults()
                  ) : (<></>)
                }
              </td>
            </tr>
          </table>
          </>) : (<></>)}
      </div>
    );
  }

  renderResults() {
    const barSize = 400;

    return (
      <>
       <table>
         {this.state.results.map(x => (
           <tr key={x.name}>
             <td className="resultName">{x.name}</td>
             <td>
                <div style={{ width: x.prob*barSize }}>
                  <div className="result">
                    { x.prob*barSize > 50 ? Math.round(x.prob*100) + '%' : (<></>)}
                  </div>
                </div>
              </td>
           </tr>
         ))}
       </table>
      </>
    );
  }
}

export default App;
