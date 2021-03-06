import {DTypeGpu, GPUTensorI} from '../../tensor/gpu/interface';
import {Operation} from './operation';

interface OpInfo<Op> {
  operation?: Op;
  numCalls: number;
  infoString: string;
}

export class Dispatcher<
  GPUTensor extends GPUTensorI,
  Info,
  Input,
  Op extends Operation<GPUTensor, Info, Input>
> {
  private opDict: {[infoString: string]: OpInfo<Op>} = {};

  constructor(
    private getOp: (dtype: DTypeGpu) => Op,
    private minCallsToCompile = 2
  ) {}

  getDefault(dtype: DTypeGpu) {
    const str = `default-${dtype}`;
    if (this.opDict[str] === undefined) {
      const op = this.getOp(dtype);
      op.compile({} as Info);

      this.opDict[str] = {
        infoString: str,
        numCalls: 0,
        operation: op,
      };
    }

    return this.opDict[str];
  }

  calc(input: Input, dtype: DTypeGpu) {
    const defaultOp = this.getDefault(dtype);

    //@ts-ignore
    const compileInfoString = defaultOp.operation.getInputInfoString(input);

    if (this.opDict[compileInfoString] === undefined) {
      this.opDict[compileInfoString] = {
        infoString: compileInfoString,
        numCalls: 0,
      };
    }
    const opInfo = this.opDict[compileInfoString];
    opInfo.numCalls++;

    if (opInfo.numCalls >= this.minCallsToCompile) {
      if (opInfo.operation === undefined) {
        opInfo.operation = this.getOp(dtype);
        //@ts-ignore
        const compileInfo = defaultOp.operation.getCompilationInfo(input);

        opInfo.operation.compile(compileInfo);
      }
      return opInfo.operation.calc(input);
    } else {
      defaultOp.numCalls++;
      //@ts-ignore
      return defaultOp.operation.calc(input);
    }
  }
}
