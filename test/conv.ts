import Tensor from '../lib/types';

import {convData} from './data/conv';

const DELTA = 0.00001;

// eslint-disable-next-line no-unused-vars
type TensorConstructor = (
  shape: ReadonlyArray<number>,
  values: number[]
) => Tensor<'float32'>;

export default function testConv(
  name: string,
  constructor: TensorConstructor,
  wait?: Promise<void>
) {
  for (let i = 0; i < convData.length; i++) {
    const dat = convData[i];
    describe(`${name} ${convData[i].name}`, () => {
      for (let j = 0; j < dat.cases.length; j++) {
        const cas = dat.cases[j];

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const args: any = dat.args ? dat.args : {};
        it(cas.name, async () => {
          if (wait) {
            await wait;
          }

          const x = constructor(cas.inputs[0].dims, cas.inputs[0].data);
          const w = constructor(cas.inputs[1].dims, cas.inputs[1].data);
          const b =
            cas.inputs.length > 2
              ? constructor(cas.inputs[2].dims, cas.inputs[2].data)
              : undefined;

          const output = constructor(cas.outputs[0].dims, cas.outputs[0].data);

          const result = x.conv(
            w,
            b,
            args['dilations'],
            args['group'],
            args['pads'],
            args['strides']
          );

          expect(await result.compare(output, DELTA)).toBeTruthy();

          x.delete();
          w.delete();
          if (b) {
            b.delete();
          }
          output.delete();
          result.delete();
        });
      }
    });
  }
}
