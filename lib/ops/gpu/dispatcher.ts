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

    const compileInfoString = defaultOp.operation.getInputInfoString(input);

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
      const compileInfo = defaultOp.operation.getCompilationInfo(input, precision);

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