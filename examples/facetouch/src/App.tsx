import React from 'react';
import './App.css';
import { loadModel } from './inference';

import * as tjs from '@hoff97/tensor-js';
import { Linear } from '@hoff97/tensor-js/dist/lib/model/basic';
import {Variable} from '@hoff97/tensor-js/dist/lib/autograd/variable';
import {bce} from '@hoff97/tensor-js/dist/lib/model/functional/bce/bce';
import {Adam} from '@hoff97/tensor-js/dist/lib/model/optimizer/adam/Adam';
import Button from '@material-ui/core/Button';
import Paper from '@material-ui/core/Paper';
import Typography from '@material-ui/core/Typography';
import LinearProgress from '@material-ui/core/LinearProgress';
import Accordion from '@material-ui/core/Accordion';
import AccordionSummary from '@material-ui/core/AccordionSummary';
import AccordionDetails from '@material-ui/core/AccordionDetails';
import Slider from '@material-ui/core/Slider';
import Grid from '@material-ui/core/Grid';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Checkbox from '@material-ui/core/Checkbox';

const numTrainSamples = 64;
const numValidationSamples = 0;
const numSamples = numTrainSamples + numValidationSamples;
//const featureDim = 64;
const featureDim = 1280;

type AppStage = 'start' | 'compiling' | 'trainData-1' | 'between-stages' | 'trainData-2' | 'training' | 'monitoring';

interface AppState {
  stage: AppStage;
  prediction?: number;
  countDown?: number;
  progress?: number;
  threshold: number;
  numIterations: number;
  sound: boolean;
  notification: boolean;
}

function wait(t: number) {
  return new Promise<void>((resolve, reject) => {
    setTimeout(() => {
      resolve();
    }, t);
  });
}

async function countDown(n: number, t: number, cb: (n: number) => void) {
  for (let i = 0; i < n; i++) {
    cb(n-i);
    await wait(t);
  }
  cb(0);
}

class App extends React.Component<{}, AppState> {
  private model?: tjs.onnx.model.OnnxModel = undefined;
  private classifier?: tjs.model.Module;
  private modelCompiled = false;

  private mean?: tjs.Tensor;
  private varSqrt?: tjs.Tensor;

  private meanMobilenet = new tjs.tensor.gpu.GPUTensor(new Float32Array([0.485, 0.456, 0.406]), [3,1,1], 16);
  private stdMobilenet = new tjs.tensor.gpu.GPUTensor(new Float32Array([0.229, 0.224, 0.225]), [3,1,1], 16);

  private data = new tjs.tensor.gpu.GPUTensor(new Float32Array(numSamples*featureDim).fill(0), [numSamples, featureDim], 32);

  private trainingResultsCollected = 0;

  private videoTensor?: tjs.tensor.gpu.GPUTensor;

  private start = Date.now().valueOf();

  private audio?: HTMLAudioElement;

  private numIts = 0;

  constructor(props: {}) {
    super(props);

    this.state = {
      stage: 'start',
      numIterations: 3,
      threshold: 0.5,
      sound: true,
      notification: false,
    };
  }

  componentWillMount() {
    loadModel('mobilenet050').then(x => {
      if (x !== undefined) {
        this.model = x;

        this.getVideo();
      }
    });

    this.audio = new Audio('alerts/beep.wav');
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
    }
  }

  prepareVideo() {
    if (this.videoTensor !== undefined) {
      const video: HTMLVideoElement = (document.querySelector("#videoElement") as any);

      this.videoTensor = tjs.tensor.gpu.GPUTensor.fromData(video, 32);

      let [height, width] = this.videoTensor.shape.slice(0,2);

      const sliced = this.videoTensor.slice([0], [3], [2]);

      this.videoTensor.delete();

      const transposed = sliced.transpose([2, 0, 1]);
      sliced.delete();

      const scaled = transposed.subtract(this.meanMobilenet);
      transposed.delete();
      const normalized = scaled.divide(this.stdMobilenet);
      scaled.delete();

      const reshaped = normalized.reshape([1,3,height,width], false);
      return reshaped;
    }
  }

  async compileModel() {
    const video: HTMLVideoElement = (document.querySelector("#videoElement") as any);
    this.videoTensor = tjs.tensor.gpu.GPUTensor.fromData(video, 32);
    if (this.videoTensor !== undefined && !this.modelCompiled) {
      this.setState({
        ...this.state,
        stage: 'compiling'
      });

      this.modelCompiled = true;

      const reshaped = this.prepareVideo() as tjs.Tensor;

      if (this.model !== undefined) {
        this.model.outputs = ["472"];
        this.model.prune(["473"]);
      }

      this.model?.optimize();

      console.log('Compiled');
      console.log('Doing forward pass');

      await countDown(3, 1000, (n: number) => {
        this.setState({
          countDown: n
        });
      });

      this.setState({
        stage: 'trainData-1',
        progress: 0
      });

      this.model?.forward([reshaped]).then(result => {
        reshaped.delete();
        this.handleResult(result);
      });
    }
  }

  processResult(tensors: tjs.Tensor[]) {
    const res1 = tensors[0];
    console.log(res1);
    return res1;
  }

  async handleResult(tensors: tjs.Tensor[]) {
    const processed = this.processResult(tensors);

    const oldResults = this.data;
    this.data = this.data.setValues(processed, [this.trainingResultsCollected, 0]) as tjs.tensor.gpu.GPUTensor;
    processed.delete();
    oldResults.delete();

    this.trainingResultsCollected++;

    console.log(`Collected ${this.trainingResultsCollected} of ${numSamples}`);

    if (this.trainingResultsCollected < numSamples/2) {
      this.setState({
        progress: this.trainingResultsCollected/(numSamples/2)*100
      });

      await wait(50);

      const reshaped = this.prepareVideo() as tjs.Tensor;

      this.model?.forward([reshaped]).then(results => {
        reshaped.delete();
        this.handleResult(results);
      });
    } else if (this.trainingResultsCollected === numSamples/2) {
      console.log('Second batch');

      this.setState({
        stage: 'between-stages'
      });

      await countDown(3, 1000, (n: number) => {
        this.setState({
          countDown: n
        });
      });

      this.setState({
        stage: 'trainData-2',
        progress: 0
      });

      const reshaped = this.prepareVideo() as tjs.Tensor;

      this.model?.forward([reshaped]).then(results => {
        reshaped.delete();
        this.handleResult(results);
      });
    } else if (this.trainingResultsCollected < numSamples) {
      this.setState({
        progress: (this.trainingResultsCollected/(numSamples/2) - 1)*100
      });

      await wait(50);
      const reshaped = this.prepareVideo() as tjs.Tensor;

      this.model?.forward([reshaped]).then(results => {
        reshaped.delete();
        this.handleResult(results);
      });
    } else {
      this.setState({
        ...this.state,
        stage: 'training',
        progress: 0
      });

      await wait(50);

      this.trainClassifier();
    }
  }

  async normalizeResults(data: tjs.Tensor) {
    this.mean = data.reduceMean(0);
    const shifted = data.subtract(this.mean);

    const variance = shifted.sumSquare(0);
    this.varSqrt = variance.sqrt();
    variance.delete();

    const normalized = shifted.divide(this.varSqrt);
    shifted.delete();
    return normalized;
  }

  async prepareData() {
    const trainData1 = this.data.slice([0],[numTrainSamples/2],[0]);
    const trainData2 = this.data.slice([numSamples/2],[numSamples/2 + numTrainSamples/2],[0]);
    let trainData = trainData1.concat(trainData2, 0);

    const valData1 = this.data.slice([numTrainSamples/2],[numSamples/2],[0]);
    const valData2 = this.data.slice([numSamples/2 + numTrainSamples/2],[numSamples],[0]);
    let valData = valData1.concat(valData2, 0);

    trainData = await this.normalizeResults(trainData);

    const shiftedValData = valData.subtract(this.mean as any);
    const normalizedValData = shiftedValData.divide(this.varSqrt as any);
    valData.delete();
    shiftedValData.delete();
    valData = normalizedValData;

    const trainX = new Variable(trainData, {noGrad: true});
    const trainYs = [...new Array(numTrainSamples/2).fill(0),...new Array(numTrainSamples/2).fill(1)]
    const trainY = Variable.create([numTrainSamples, 1], new Float32Array(trainYs), 'GPU', {noGrad: true});

    const valX = new Variable(valData, {noGrad: true});
    const valYs = [...new Array(numValidationSamples/2).fill(0),...new Array(numValidationSamples/2).fill(1)]

    return {trainX, trainYs, trainY, valX, valYs};
  }

  async trainClassifier() {
    this.classifier = new Linear(featureDim, 1, true);
    await this.classifier.toGPU(32);

    const {trainX, trainYs, trainY, valX, valYs} = await this.prepareData();

    const optimizer = new Adam(this.classifier);

    const numIts = 200;

    for (let i = 0; i < numIts; i++) {
      this.setState({
        ...this.state,
        progress: i/numIts*100
      });

      await wait(10);

      const pred = (await this.classifier.forward([trainX]))[0];
      const sigmoid = pred.sigmoid();

      const loss = bce(sigmoid, trainY).reduceMean() as Variable;

      if (i % 50 === 0) {
        console.log(i, (await loss.getValues())[0]);

        const predTrain = Array.from(await sigmoid.getValues());
        let correctTrain = this.getCorrect(predTrain, trainYs);
        console.log(`${correctTrain} of ${predTrain.length} predicted correctly`);


        const predVal = (await this.classifier.forward([valX]))[0];
        const sigmoidVal = predVal.sigmoid();
        const predValArr = Array.from(await sigmoidVal.getValues());
        let correctVal = this.getCorrect(predValArr, valYs);
        console.log(`${correctVal} of ${predValArr.length} predicted correctly`);
        sigmoidVal.delete();
      }

      loss.backward();
      optimizer.step();

      loss.delete();
      optimizer.zeroGrads();
    }

    this.setState({
      ...this.state,
      stage: 'monitoring',
      prediction: 0
    });

    await wait(1000);

    const reshaped = this.prepareVideo() as tjs.Tensor;

    this.model?.forward([reshaped]).then(results => {
      reshaped.delete();
      this.process(results);
    });
  }

  toggleNotification(value: boolean) {
    if (value) {
      if (Notification.permission !== "denied" && Notification.permission !== "granted") {
        Notification.requestPermission().then((permission) => {
          if (permission === "granted") {
            new Notification("I will notify you like this.");
            this.setState({
              notification: true
            });
          }
        });
      } else if (Notification.permission === "granted") {
        this.setState({
          notification: true
        });
      }
    } else {
      this.setState({
        notification: false
      });
    }
  }

  async process(tensors: tjs.Tensor[]) {
    const processed = this.processResult(tensors);

    const shifted = processed.subtract(this.mean as any);
    processed.delete();
    const normalized = new Variable(shifted.divide(this.varSqrt as any), {noGrad: true});
    shifted.delete();

    //@ts-ignore
    const logits = (await this.classifier.forward([normalized]))[0];
    normalized.delete();
    const v = await logits.getValues();
    logits.delete();

    const sigmoid = 1/(1+Math.exp(-v[0]));

    this.setState({
      ...this.state,
      prediction: sigmoid
    });

    if (sigmoid > this.state.threshold) {
      this.numIts++;
      if (this.numIts === this.state.numIterations) {
        if (this.state.sound) {
          this.audio?.play();
        }
        if (this.state.notification) {
          new Notification('Stop touching your face');
        }
      }
    } else {
      this.numIts = 0;
    }

    await wait(1000);

    const reshaped = this.prepareVideo() as tjs.Tensor;

    this.model?.forward([reshaped]).then(results => {
      reshaped.delete();
      this.process(results);
    });
  }

  getCorrect(predVals: number[], ys: number[]) {
    let correct = 0;
    for (let i = 0; i <predVals.length; i++) {
      if ((ys[i] === 1 && predVals[i] > 0.5) || (ys[i] === 0 && predVals[i] < 0.5)) {
        correct++;
      }
    }
    return correct;
  }

  render() {
    return (
      <div className="App">
        <Paper elevation={3} style={{padding: '10px'}}>
         {this.renderSettings()}

          <Typography variant="h3" component="h2">
            Dont touch your face
          </Typography>
          This app will teach you not to touch your face.<br/>

          <div className="mainContent">
            <video autoPlay id="videoElement"/><br/>
            {this.renderState()}
          </div>
        </Paper>
      </div>
    );
  }

  renderSettings() {
    return (
      <Accordion className="settings">
        <AccordionSummary
          aria-controls="panel1a-content"
          id="panel1a-header"
        >
          <Typography variant="h5">Settings</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={1}>
            <Grid item xs={12}>
              <Typography id="slider-threshold-label" gutterBottom>
                Threshold
              </Typography><div></div>
            </Grid>
            <Grid item xs={12}>
              <Slider
                aria-labelledby="slider-threshold-label"
                value={this.state.threshold}
                min={0}
                max={1}
                step={0.1}
                onChange={(ev, v) => this.setState({threshold: v as number})}
                valueLabelDisplay="auto"
              />
            </Grid>
            <Grid item xs={12}>
              <Typography id="slider-threshold-label" gutterBottom>
                Seconds until alert
              </Typography><div></div>
            </Grid>
            <Grid item xs={12}>
              <Slider
                aria-labelledby="slider-threshold-label"
                value={this.state.numIterations}
                min={1}
                max={10}
                step={1}
                onChange={(ev, v) => this.setState({numIterations: v as number})}
                valueLabelDisplay="auto"
              />
            </Grid>
            <Grid item xs={6}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={this.state.sound}
                    onChange={(ev) => this.setState({sound: ev.target.checked})}
                    name="checkedB"
                    color="primary"
                  />
                }
                label="Play sound"
              />
            </Grid>
            <Grid item xs={6}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={this.state.notification}
                    onChange={(ev) => this.toggleNotification(ev.target.checked)}
                    name="checkedB"
                    color="primary"
                  />
                }
                label="Show notification"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>
    );
  }

  renderState() {
    if (this.state.stage === 'start') {
      return (<Button variant="contained" color="primary" onClick={() => this.compileModel()}>Start</Button>);
    } else if (this.state.stage === 'compiling') {
      return (<> Dont touch your face until the bar is full. Starting in {this.state.countDown} seconds. </>);
    } else if (this.state.stage === 'trainData-1' || this.state.stage === 'trainData-2') {
      return (<LinearProgress variant="determinate" value={this.state.progress} className="progress"/>)
    } else if (this.state.stage === 'between-stages') {
      return (<> Next, touch your face until the bar is full. Starting in {this.state.countDown} seconds. </>);
    } else if (this.state.stage === 'training') {
      return (
        <div> Done. Training face touch recognition model
          <LinearProgress variant="indeterminate" className="progress"/>
        </div>
      );
    } else if (this.state.stage === 'monitoring') {
      const pred = (this.state.prediction as number);

      const maxColor = 200;
      const color = `rgb(${Math.round(pred*maxColor)},${maxColor-Math.round(pred*maxColor)},0)`;

      return (
        <div> Done. I will alert you when i think you are touching your face
          <div style={{
              width: 320,
              height: 20,
              backgroundColor: '#AAA',
              borderRadius: 5,
              overflow: 'hidden',
              margin: 'auto'
          }}>
            <div style={{
              width: pred*320,
              height: 20,
              backgroundColor: color
            }}></div>
          </div>
        </div>
      );
    }
  }
}

export default App;
