import React from 'react';
import './App.css';

import Plot from 'react-plotly.js';
import { Linear, Relu, Sequential } from '@hoff97/tensor-js/dist/lib/model/basic';
import { Adam } from '@hoff97/tensor-js/dist/lib/model/optimizer/adam/Adam';
import { Variable } from '@hoff97/tensor-js/dist/lib/autograd/variable';
import { SGD } from '@hoff97/tensor-js/dist/lib/model/optimizer/SGD';
import { Button, FormControl, Grid, InputLabel, MenuItem, Select, Slider, Typography } from '@material-ui/core';
import LinearProgress from '@material-ui/core/LinearProgress';
import { Data } from 'plotly.js';

interface Curve {
  xs: number[];
  ys: number[];
  name: string;
}

interface AppState {
  curves: Curve[];
  losses: Curve[];
  noise: number;
  numSamples: number;
  numIterations: number;
  function: number;
  trainProgress?: number;
  trainData?: Curve;
  valData?: Curve;
  optimizer: 'Adam' | 'SGD';
}

const functions = [
  {
    name: 'Cosine',
    start: 0,
    end: 2*Math.PI,
    f: (x: number) => Math.cos(x)
  },
  {
    name: 'Cubic Polynomial',
    start: -2*Math.PI,
    end: 2*Math.PI,
    f: (x: number) => 3*x*x*x - 2*x*x
  },
  {
    name: 'Sine',
    start: 0,
    end: 2*Math.PI,
    f: (x: number) => Math.sin(x)
  }
];

class App extends React.Component<{}, AppState> {
  constructor(props: {}) {
    super(props);

    this.state = {
      curves: [],
      losses: [],
      noise: 0.01,
      optimizer: 'Adam',
      numSamples: 100,
      numIterations: 1000,
      function: 0
    };
  }

  uniform(start: number, end: number, n: number) {
    const xs = [];
    for (let i = 0; i<n; i++) {
      const x = i/n*(end - start) + start;

      xs.push(x);
    }
    return xs;
  }

  random(start: number, end: number, n: number) {
    let xs = [];
    for (let i = 0; i<n; i++) {
      const x = Math.random()*(end-start) + start;

      xs.push(x);
    }
    xs = xs.sort();
    return xs;
  }

  getFunctionValue(x: number) {
    return functions[this.state.function].f(x);
  }

  getFunctionRange() {
    return {
      start: functions[this.state.function].start,
      end: functions[this.state.function].end,
    };
  }

  generateYs(xs: number[], noise: number = 0) {
    console.log(noise);
    const ys = [];
    for (let i = 0; i<xs.length; i++) {
      const y = this.getFunctionValue(xs[i]) + Math.random()*noise - noise/2;

      ys.push(y);
    }
    return ys;
  }

  async doStuff() {
    this.setState({
      ...this.state,
      curves: [],
      trainData: undefined
    });

    const l1 = new Linear(1, 128);
    const l2 = new Relu();
    const l3 = new Linear(128, 128);
    const l4 = new Relu();
    const l5 = new Linear(128, 64);
    const l6 = new Relu();
    const l7 = new Linear(64, 1);

    const backend = 'GPU';

    const model = new Sequential([l1, l2, l3, l4, l5, l6, l7]);
    await model.toBackend(backend);

    let optimizer;
    if (this.state.optimizer === 'Adam') {
      optimizer = new Adam(model);
    } else {
      optimizer = new SGD(model, 0.5);
    }

    const {start, end} = this.getFunctionRange();
    const n = this.state.numSamples;
    const xs = this.uniform(start, end, n);
    const ys = this.generateYs(xs);

    const curves = this.state.curves || [];

    this.setState({
      ...this.state,
      curves: [...curves, {
        name: 'Ground truth',
        xs: xs,
        ys: ys
      }]
    });

    let xTrain = this.random(start, end, n);
    let yTrain = this.generateYs(xTrain, this.state.noise);

    const valFraction = 5;
    let xVal = this.random(start, end, n/valFraction);
    let yVal = this.generateYs(xVal, this.state.noise);

    this.setState({
      ...this.state,
      trainData: {
        name: 'Train Data',
        xs: xTrain,
        ys: yTrain
      },
      valData: {
        name: 'Validation Data',
        xs: xVal,
        ys: yVal
      }
    })

    let xTrainVar = Variable.create([n,1], xTrain, backend, {noGrad: true});
    let yTrainVar = Variable.create([n, 1], yTrain, backend, {noGrad: true});
    let xValVar = Variable.create([n/valFraction,1], xVal, backend, {noGrad: true});
    let yValVar = Variable.create([n/valFraction, 1], yVal, backend, {noGrad: true});

    await new Promise((resolve, reject) => {
      setTimeout(() => {
        resolve(0);
      }, 1000);
    });

    const startTime = Date.now();

    const its = [];
    const trainLoss = [];
    const valLoss = [];

    const reportEvery = 20;

    for (let i = 0; i < this.state.numIterations; i++) {
      const res = (await model.forward([xTrainVar]))[0];

      const diff = res.subtract(yTrainVar);
      const loss = diff.reduceMeanSquare() as Variable;

      loss.backward();
      optimizer.step();

      if(i%reportEvery === 0) {
        const lossV = (await loss.value.getValues())[0];
        trainLoss.push(lossV);
        its.push(i);
      }

      loss.delete();
      optimizer.zeroGrads();

      if (i%reportEvery === 0) {
        const resVal = (await model.forward([xValVar]))[0];

        const diff = resVal.subtract(yValVar);
        const loss = diff.reduceMeanSquare() as Variable;

        const lossValidation = (await loss.value.getValues())[0];
        valLoss.push(lossValidation);

        loss.delete();
        optimizer.zeroGrads();

        this.setState({
          ...this.state,
          trainProgress: i/this.state.numIterations*100
        });
        await new Promise((resolve, reject) => {
          setTimeout(() => {
            resolve(0);
          }, 50);
        });
      }
    }
    const endTime = Date.now();

    console.log(`Done, took ${(endTime - startTime)}ms`);

    this.setState({
      ...this.state,
      trainProgress: undefined,
      losses: [
        {
          name: 'Train loss',
          xs: its,
          ys: trainLoss
        },
        {
          name: 'Validation loss',
          xs: its,
          ys: valLoss
        }
      ]
    });

    const pred = await (await model.forward([xTrainVar]))[0].getValues();
    this.setState({
      ...this.state,
      curves: [...this.state.curves, {
        name: 'Predicition',
        xs: xs,
        ys: Array.from(pred)
      }]
    });
  }

  render() {
    return (
      <div className="App">
        <h1>Training visualization</h1>
        {this.renderSettings()}
        <Button color="primary" onClick={() => this.doStuff()}>Run training</Button><br/>

        {this.state.trainProgress !== undefined ? (
          <LinearProgress value={this.state.trainProgress} variant='determinate'/>
        ) : (<></>)}

        <Grid item xs={12}>
          <Grid container justify="center" spacing={2}>
            <Grid item>
              {this.renderValuePlot()}
            </Grid>
            <Grid item>
              {this.renderLossPlot()}
            </Grid>
          </Grid>
        </Grid>
      </div>
    );
  }

  renderValuePlot() {
    let curves: Curve[] = this.state.curves;

    const plotData: Data[] = curves.map(curve => { return {
      x: curve.xs,
      y: curve.ys,
      type: 'scatter',
      mode: 'lines',
      name: curve.name
    };});

    if (this.state.trainData !== undefined) {
      plotData.push({
        x: this.state.trainData.xs,
        y: this.state.trainData.ys,
        type: 'scatter',
        mode: 'markers',
        name: 'Training data'
      })
    }
    if (this.state.valData !== undefined) {
      plotData.push({
        x: this.state.valData.xs,
        y: this.state.valData.ys,
        type: 'scatter',
        mode: 'markers',
        name: 'Validation data'
      })
    }

    return (
      <Plot
        data={plotData as any}
        layout={ {width: 600, height: 400, title: 'Ground truth and learned curve'} }
        config={{
          showTips: false,
          displaylogo: false,
          modeBarButtons: false
        }}
      />
    );
  }

  renderLossPlot() {
    let curves: Curve[] = this.state.losses;

    const plotData: Data[] = curves.map(curve => { return {
      x: curve.xs,
      y: curve.ys,
      type: 'scatter',
      mode: 'lines',
      name: curve.name
    };});

    return (
      <Plot
        data={plotData as any}
        layout={{
          width: 600, height: 400, title: 'Losses',
          xaxis: {
            autorange: true
          },
          yaxis: {
            type: 'log',
            autorange: true
          }
        }}
        config={{
          showTips: false,
          displaylogo: false,
          modeBarButtons: false,
        }}
      />
    );
  }

  renderSettings() {
    let noise = this.state.noise;

    return (
      <Grid container spacing={2}>
          <Grid item xs>
            <FormControl>
              <InputLabel id="label-function">Function</InputLabel>
              <Select
                labelId="label-function"
                id="select-function"
                value={this.state.function}
                onChange={(ev, newValue) => this.setState({...this.state, function: ev.target.value as any})}
              >
                {functions.map((f, i) => (<MenuItem key={i} value={i}>{f.name}</MenuItem>))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs>
            <Typography id="slide-noise" gutterBottom>
              Noise in training data
            </Typography>
            <Slider value={noise}
              onChange={(ev, newValue) => this.setState({...this.state, noise: newValue as number})}
              aria-labelledby="slider-noise"
              min={0}
              max={0.4}
              step={0.01}
              marks
              valueLabelDisplay="auto"/>
          </Grid>
          <Grid item xs>
            <Typography id="slide-n-samples" gutterBottom>
              Number of samples
            </Typography>
            <Slider value={this.state.numSamples}
              onChange={(ev, newValue) => this.setState({...this.state, numSamples: newValue as number})}
              aria-labelledby="slider-n-samples"
              min={50}
              max={400}
              step={50}
              marks
              valueLabelDisplay="auto"/>
          </Grid>
          <Grid item xs>
            <Typography id="slide-n-iterations" gutterBottom>
              Number of iterations for training
            </Typography>
            <Slider value={this.state.numIterations}
              onChange={(ev, newValue) => this.setState({...this.state, numIterations: newValue as number})}
              aria-labelledby="slider-n-samples"
              min={200}
              max={2000}
              step={200}
              marks
              valueLabelDisplay="auto"/>
          </Grid>
          <Grid item xs>
            <FormControl>
              <InputLabel id="label-optimizer">Optimizer</InputLabel>
              <Select
                labelId="label-optimizer"
                id="demo-simple-select-helper"
                value={this.state.optimizer}
                onChange={(ev, newValue) => this.setState({...this.state, optimizer: ev.target.value as any})}
              >
                <MenuItem value='Adam'>Adam</MenuItem>
                <MenuItem value='SGD'>SGD</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
    );
  }
}

export default App;
