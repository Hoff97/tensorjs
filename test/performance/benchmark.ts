import CPUTensor from "../../lib/tensor/cpu/tensor";
import GPUTensor from "../../lib/tensor/gpu/tensor";
import WASMTensor, { wasmLoaded } from "../../lib/tensor/wasm/tensor";
import Tensor from "../../lib/types";

declare const suite: any;
declare const benchmark: any;

const cpuConstructor = (shape: ReadonlyArray<number>, values: number[]) => new CPUTensor(shape, values);
const gpuConstructor = (shape: ReadonlyArray<number>, values: number[]) => {
  const vals = Float32Array.from(values);
  return new GPUTensor(vals, shape);
};
const wasmConstructor = (shape: ReadonlyArray<number>, values: number[]) => {
  const sh = Uint32Array.from(shape);
  const vals = Float32Array.from(values);
  return new WASMTensor(vals, sh);
};

// TODO: Find a way to use wasm
const backends = [
  { name: 'CPU', constructor: cpuConstructor },
  { name: 'WASM', constructor: wasmConstructor },
  { name: 'GPU', constructor: gpuConstructor },
];

function randomValues(length: number) {
  const values: number[] = [];
  for (let i = 0; i < length; i++) {
    values.push(Math.random());
  }
  return values;
}

suite("Tensor create", () => {
  const values = randomValues(100*100);
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const tensor = backend.constructor([100,100], values);
      tensor.delete();
    });
  }
});

let utility = {};

suite("Tensor exp", () => {
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const result = (utility as any).tensors[backend.name].exp();

      result.delete();
    });
  }
}, {
  onStart() {
    const values = randomValues(100*100);

    const tensors: {[name: string]: Tensor} = {};
    for (let backend of backends) {
      tensors[backend.name] = backend.constructor([100,100], values);
    }

    (utility as any).tensors = tensors;
  },
  onComplete() {
    for (let backend of backends) {
      (utility as any).tensors[backend.name].delete();
    }
    utility = {};
  }
});

suite("Tensor log", () => {
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const result = (utility as any).tensors[backend.name].log();
      result.delete();
    });
  }
}, {
  onStart() {
    const values = randomValues(100*100);

    const tensors: {[name: string]: Tensor} = {};
    for (let backend of backends) {
      tensors[backend.name] = backend.constructor([100,100], values);
    }

    (utility as any).tensors = tensors;
  },
  onComplete() {
    for (let backend of backends) {
      (utility as any).tensors[backend.name].delete();
    }
    utility = {};
  }
});

suite("Tensor sqrt", () => {
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const result = (utility as any).tensors[backend.name].sqrt();
      result.delete();
    });
  }
}, {
  onStart() {
    const values = randomValues(100*100);

    const tensors: {[name: string]: Tensor} = {};
    for (let backend of backends) {
      tensors[backend.name] = backend.constructor([100,100], values);
    }

    (utility as any).tensors = tensors;
  },
  onComplete() {
    for (let backend of backends) {
      (utility as any).tensors[backend.name].delete();
    }
    utility = {};
  }
});

suite("Tensor abs", () => {
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const result = (utility as any).tensors[backend.name].abs();
      result.delete();
    });
  }
}, {
  onStart() {
    const values = randomValues(100*100);

    const tensors: {[name: string]: Tensor} = {};
    for (let backend of backends) {
      tensors[backend.name] = backend.constructor([100,100], values);
    }

    (utility as any).tensors = tensors;
  },
  onComplete() {
    for (let backend of backends) {
      (utility as any).tensors[backend.name].delete();
    }
    utility = {};
  }
});

suite("Tensor transpose", () => {
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const result = (utility as any).tensors[backend.name].transpose();
      result.delete();
    });
  }
}, {
  onStart() {
    const values = randomValues(20*30*40);

    const tensors: {[name: string]: Tensor} = {};
    for (let backend of backends) {
      tensors[backend.name] = backend.constructor([20,30,40], values);
    }

    (utility as any).tensors = tensors;
  },
  onComplete() {
    for (let backend of backends) {
      (utility as any).tensors[backend.name].delete();
    }
    utility = {};
  }
});

suite("Tensor add", () => {
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const t1: Tensor = (utility as any).tensors[backend.name][0];
      const t2: Tensor = (utility as any).tensors[backend.name][1];

      const result = t1.add(t2);
      result.delete();
    });
  }
}, {
  onStart() {
    const values1 = randomValues(100*100);
    const values2 = randomValues(100*100);

    const tensors: {[name: string]: Tensor[]} = {};
    for (let backend of backends) {
      tensors[backend.name] = []
      tensors[backend.name].push(backend.constructor([100,100], values1));
      tensors[backend.name].push(backend.constructor([100,100], values2));
    }

    (utility as any).tensors = tensors;
  },
  onComplete() {
    for (let backend of backends) {
      for (let tensor of (utility as any).tensors[backend.name]) {
        tensor.delete();
      }
    }
    utility = {};
  }
});

suite("Tensor subtract", () => {
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const t1: Tensor = (utility as any).tensors[backend.name][0];
      const t2: Tensor = (utility as any).tensors[backend.name][1];

      const result = t1.subtract(t2);
      result.delete();
    });
  }
}, {
  onStart() {
    const values1 = randomValues(100*100);
    const values2 = randomValues(100*100);

    const tensors: {[name: string]: Tensor[]} = {};
    for (let backend of backends) {
      tensors[backend.name] = []
      tensors[backend.name].push(backend.constructor([100,100], values1));
      tensors[backend.name].push(backend.constructor([100,100], values2));
    }

    (utility as any).tensors = tensors;
  },
  onComplete() {
    for (let backend of backends) {
      for (let tensor of (utility as any).tensors[backend.name]) {
        tensor.delete();
      }
    }
    utility = {};
  }
});

suite("Tensor divide", () => {
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const t1: Tensor = (utility as any).tensors[backend.name][0];
      const t2: Tensor = (utility as any).tensors[backend.name][1];

      const result = t1.divide(t2);
      result.delete();
    });
  }
}, {
  onStart() {
    const values1 = randomValues(100*100);
    const values2 = randomValues(100*100);

    const tensors: {[name: string]: Tensor[]} = {};
    for (let backend of backends) {
      tensors[backend.name] = []
      tensors[backend.name].push(backend.constructor([100,100], values1));
      tensors[backend.name].push(backend.constructor([100,100], values2));
    }

    (utility as any).tensors = tensors;
  },
  onComplete() {
    for (let backend of backends) {
      for (let tensor of (utility as any).tensors[backend.name]) {
        tensor.delete();
      }
    }
    utility = {};
  }
});

suite("Tensor multiply", () => {
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const t1: Tensor = (utility as any).tensors[backend.name][0];
      const t2: Tensor = (utility as any).tensors[backend.name][1];

      const result = t1.multiply(t2);
      result.delete();
    });
  }
}, {
  onStart() {
    const values1 = randomValues(100*100);
    const values2 = randomValues(100*100);

    const tensors: {[name: string]: Tensor[]} = {};
    for (let backend of backends) {
      tensors[backend.name] = []
      tensors[backend.name].push(backend.constructor([100,100], values1));
      tensors[backend.name].push(backend.constructor([100,100], values2));
    }

    (utility as any).tensors = tensors;
  },
  onComplete() {
    for (let backend of backends) {
      for (let tensor of (utility as any).tensors[backend.name]) {
        tensor.delete();
      }
    }
    utility = {};
  }
});

suite("Tensor power", () => {
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const t1: Tensor = (utility as any).tensors[backend.name][0];
      const t2: Tensor = (utility as any).tensors[backend.name][1];

      const result = t1.power(t2);
      result.delete();
    });
  }
}, {
  onStart() {
    const values1 = randomValues(100*100);
    const values2 = randomValues(100*100);

    const tensors: {[name: string]: Tensor[]} = {};
    for (let backend of backends) {
      tensors[backend.name] = []
      tensors[backend.name].push(backend.constructor([100,100], values1));
      tensors[backend.name].push(backend.constructor([100,100], values2));
    }

    (utility as any).tensors = tensors;
  },
  onComplete() {
    for (let backend of backends) {
      for (let tensor of (utility as any).tensors[backend.name]) {
        tensor.delete();
      }
    }
    utility = {};
  }
});

suite("Tensor matmul", () => {
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const result = (utility as any).tensors[backend.name][0].matMul((utility as any).tensors[backend.name][1]);
      result.delete();
    });
  }
}, {
  onStart() {
    const size = 100;
    
    const values1 = randomValues(size*size);
    const values2 = randomValues(size*size);

    const tensors: {[name: string]: Tensor[]} = {};
    for (let backend of backends) {
      tensors[backend.name] = [];
      tensors[backend.name].push(backend.constructor([size,size], values1));
      tensors[backend.name].push(backend.constructor([size,size], values2));
    }
    (utility as any).tensors = tensors;
  },
  onComplete() {
    for (let backend of backends) {
      for (let tensor of (utility as any).tensors[backend.name]) {
        tensor.delete();
      }
    }
    utility = {};
  }
});

for (let axis of [[1,2],[0,1,2,3]]) {
  suite(`Tensor sum ${axis}`, () => {
    for (let backend of backends) {
      benchmark(backend.name, () => {
        const tensor: Tensor = (utility as any).tensors[backend.name];
        const result = tensor.sum(axis)
        result.delete();
      });
    }
  }, {
    onStart() {
      const size = 10;
      
      const values1 = randomValues(size*size*size*size);
  
      const tensors: {[name: string]: Tensor} = {};
      for (let backend of backends) {
        tensors[backend.name] = backend.constructor([size,size,size,size], values1);
      }
      (utility as any).tensors = tensors;
    },
    onComplete() {
      for (let backend of backends) {
        (utility as any).tensors[backend.name].delete();
      }
      utility = {};
    }
  });
}

for (let axis of [[1,2],[0,1,2,3]]) {
  suite(`Tensor product ${axis}`, () => {
    for (let backend of backends) {
      benchmark(backend.name, () => {
        const tensor: Tensor = (utility as any).tensors[backend.name];
        const result = tensor.product(axis)
        result.delete();
      });
    }
  }, {
    onStart() {
      const size = 10;
      
      const values1 = randomValues(size*size*size*size);
  
      const tensors: {[name: string]: Tensor} = {};
      for (let backend of backends) {
        tensors[backend.name] = backend.constructor([size,size,size,size], values1);
      }
      (utility as any).tensors = tensors;
    },
    onComplete() {
      for (let backend of backends) {
        (utility as any).tensors[backend.name].delete();
      }
      utility = {};
    }
  });
}

for (let axis of [[1,2],[0,1,2,3]]) {
  suite(`Tensor max ${axis}`, () => {
    for (let backend of backends) {
      benchmark(backend.name, () => {
        const tensor: Tensor = (utility as any).tensors[backend.name];
        const result = tensor.max(axis)
        result.delete();
      });
    }
  }, {
    onStart() {
      const size = 10;
      
      const values1 = randomValues(size*size*size*size);
  
      const tensors: {[name: string]: Tensor} = {};
      for (let backend of backends) {
        tensors[backend.name] = backend.constructor([size,size,size,size], values1);
      }
      (utility as any).tensors = tensors;
    },
    onComplete() {
      for (let backend of backends) {
        (utility as any).tensors[backend.name].delete();
      }
      utility = {};
    }
  });
}

for (let axis of [[1,2],[0,1,2,3]]) {
  suite(`Tensor min ${axis}`, () => {
    for (let backend of backends) {
      benchmark(backend.name, () => {
        const tensor: Tensor = (utility as any).tensors[backend.name];
        const result = tensor.min(axis)
        result.delete();
      });
    }
  }, {
    onStart() {
      const size = 10;
      
      const values1 = randomValues(size*size*size*size);
  
      const tensors: {[name: string]: Tensor} = {};
      for (let backend of backends) {
        tensors[backend.name] = backend.constructor([size,size,size,size], values1);
      }
      (utility as any).tensors = tensors;
    },
    onComplete() {
      for (let backend of backends) {
        (utility as any).tensors[backend.name].delete();
      }
      utility = {};
    }
  });
}

suite("Tensor conv", () => {
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const x: Tensor = (utility as any).tensors[backend.name][0];
      const w: Tensor = (utility as any).tensors[backend.name][1];
      const result = x.conv(w, undefined, [1,1], 1, [0,0,0,0], [1,1]);
      
      result.delete();
    });
  }
}, {
  onStart() {
    const x = randomValues(1*8*30*30);
    const w = randomValues(4*8*5*5);

    const tensors: {[name: string]: Tensor[]} = {};
    for (let backend of backends) {
      tensors[backend.name] = [];
      tensors[backend.name].push(backend.constructor([1,8,30,30], x));
      tensors[backend.name].push(backend.constructor([4,8,5,5], w));
    }

    (utility as any).tensors = tensors;
  },
  onComplete() {
    for (let backend of backends) {
      for (let tensor of (utility as any).tensors[backend.name]) {
        tensor.delete();
      }
    }
    utility = {};
  }
});

suite("Tensor softmax", () => {
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const x: Tensor = (utility as any).tensors[backend.name][0];
      const result = x.softmax(1);

      result.delete();
    });
  }
}, {
  onStart() {
    const x = randomValues(100*100);

    const tensors: {[name: string]: Tensor[]} = {};
    for (let backend of backends) {
      tensors[backend.name] = [];
      tensors[backend.name].push(backend.constructor([100,100], x));
    }

    (utility as any).tensors = tensors;
  },
  onComplete() {
    for (let backend of backends) {
      for (let tensor of (utility as any).tensors[backend.name]) {
        tensor.delete();
      }
    }
    utility = {};
  }
});

suite("Tensor gemm", () => {
  for (let backend of backends) {
    benchmark(backend.name, () => {
      const a: Tensor = (utility as any).tensors[backend.name][0];
      const b: Tensor = (utility as any).tensors[backend.name][0];
      const c: Tensor = (utility as any).tensors[backend.name][0];
      const result = a.gemm(b, false, false, 1, c, 1);

      result.delete();
    });
  }
}, {
  onStart() {
    const a = randomValues(2*100*100);
    const b = randomValues(2*100*50);
    const c = randomValues(100*50);

    const tensors: {[name: string]: Tensor[]} = {};
    for (let backend of backends) {
      tensors[backend.name] = [];
      tensors[backend.name].push(backend.constructor([2,100,100], a));
      tensors[backend.name].push(backend.constructor([2,100,50], b));
      tensors[backend.name].push(backend.constructor([100,50], c));
    }

    (utility as any).tensors = tensors;
  },
  onComplete() {
    for (let backend of backends) {
      for (let tensor of (utility as any).tensors[backend.name]) {
        tensor.delete();
      }
    }
    utility = {};
  }
});