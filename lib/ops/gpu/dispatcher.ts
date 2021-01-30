import { GPUTensorI } from "../../tensor/gpu/interface";
import { Precision } from "../../types";
import { Operation } from "./operation";

interface OpInfo<Op> {
  operation?: Op,
  numCalls: number;
  infoString: string;
}

export class Dispatcher<GPUTensor extends GPUTensorI, Info, Input, Op extends Operation<GPUTensor, Info, Input>> {
  private opDict: {[infoString: string]: OpInfo<Op>} = {};

  constructor(private getOp: () => Op, private minCallsToCompile = 2) {
  }

  compileInfoToString(info: Info, precision: Precision) {
    // TODO: Is this fast enough?
    return JSON.stringify(info) + `-${precision}`;
  }

  getDefault(precision: Precision) {
    const str = `default-${precision}`;
    if (this.opDict[str] === undefined) {
      const op = this.getOp();
      op.compile({} as any, precision);

      this.opDict[str] = {
        infoString: str,
        numCalls: 0,
        operation: op
      };
    }

    return this.opDict[str];
  }

  calc(input: Input, precision: Precision) {
    const defaultOp = this.getDefault(precision);

    const compileInfo = defaultOp.operation.getCompilationInfo(input, precision);
    const compileInfoString = this.compileInfoToString(compileInfo, precision);

    if (this.opDict[compileInfoString] === undefined) {
      this.opDict[compileInfoString] = {
        infoString: compileInfoString,
        numCalls: 0
      };
    }
    const opInfo = this.opDict[compileInfoString];
    opInfo.numCalls++;

    if (opInfo.numCalls === this.minCallsToCompile) {
      opInfo.operation = this.getOp();
      console.log('Compiling', compileInfo, compileInfoString);
      opInfo.operation.compile(compileInfo, precision);
    }

    if (opInfo.numCalls >= this.minCallsToCompile) {
      return opInfo.operation.calc(input);
    } else {
      defaultOp.numCalls++;
      return defaultOp.operation.calc(input);
    }
  }
}