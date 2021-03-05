use crate::shape::*;
use crate::tensor::*;
use crate::utils::conv_output_size;
use crate::utils::conv_transpose_output_size;
use js_sys::Float32Array;
use js_sys::Uint32Array;
use num_traits::zero;
use num_traits::Float;
use num_traits::FromPrimitive;
use num_traits::Num;

impl<DType> Tensor<DType>
where
    DType: Copy,
    DType: Num,
    DType: FromPrimitive,
    DType: PartialOrd,
{
    pub fn _conv(
        &self,
        kernel: &Tensor<DType>,
        bias: Option<&Tensor<DType>>,
        _dilations: &Vec<usize>,
        group: usize,
        _pads: &Vec<usize>,
        _strides: &Vec<usize>,
        activation: u32,
    ) -> Tensor<DType> {
        let N = self.get_dim_size(0);
        let C = self.get_dim_size(1);
        let D = self.get_sh();
        let W = kernel.get_sh();
        let M = kernel.get_dim_size(0);
        let CG = kernel.get_dim_size(1);

        let kernel_size = get_size_from(W, 2);

        let data_rank = self.rank() - 2;

        let R = conv_output_size(D, W, _pads, _dilations, _strides, 2);
        let output_size = get_size(&R);

        let mut output_shape = vec![0; data_rank + 2];
        output_shape[0] = N;
        output_shape[1] = M;
        for i in 0..data_rank {
            output_shape[i + 2] = R[i];
        }

        let output_strides = compute_strides(&output_shape);
        let o_size = get_size(&output_shape);
        let mut values = vec![zero(); o_size];

        // Iterate over all batches
        for n in 0..N {
            // Iterate over all output channels
            for m in 0..M {
                let basis = output_strides[0] * n + output_strides[1] * m;

                match bias {
                    None => {}
                    Some(bi) => {
                        let b = bi.get_ix(m);

                        for o in 0..output_size {
                            values[basis + o] = b;
                        }
                    }
                }

                for cg in 0..CG {
                    let c = (m * CG + cg) % C;

                    let mut output_indices = vec![0; data_rank + 2];
                    output_indices[0] = n;
                    output_indices[1] = m;

                    for o_ix in 0..output_size {
                        let mut result = values[basis + o_ix];

                        let mut kernel_indices = vec![0; data_rank + 2];
                        kernel_indices[0] = m;
                        kernel_indices[1] = cg;
                        let kernel_base =
                            m * kernel.get_strides_at(0) + cg * kernel.get_strides_at(1);

                        for kernel_ix in 0..kernel_size {
                            let mut input_ix = vec![0; data_rank + 2];
                            input_ix[0] = n;
                            input_ix[1] = c;

                            let mut skip = false;
                            for axis in 0..data_rank {
                                let ix = (output_indices[axis + 2] * _strides[axis]) as i32
                                    + (kernel_indices[axis + 2] * _dilations[axis]) as i32
                                    - (_pads[axis] as i32);

                                if ix < 0 || ix >= D[axis + 2] as i32 {
                                    skip = true;
                                    break;
                                }

                                input_ix[axis + 2] = ix as usize;
                            }

                            if !skip {
                                let w_i = kernel.get_ix(kernel_base + kernel_ix);
                                let x_i = self.get(&input_ix);
                                result = result + w_i * x_i;
                            }

                            increment_index(&mut kernel_indices, kernel.get_sh());
                        }

                        values[basis + o_ix] = result;

                        increment_index(&mut output_indices, &output_shape);
                    }
                }

                if activation != 0 {
                    for o in 0..output_size {
                        let mut result = values[basis + o];
                        if activation == 1 {
                            if result < zero() {
                                result = zero()
                            }
                        } else if activation == 2 {
                            match DType::from_u32(6) {
                                Some(six) => if result < zero() {
                                    result = zero()
                                } else if result > six {
                                    result = six
                                },
                                None => panic!("Encountered dtype that can not represent 6 for relu6 activation")
                            }
                        }
                        values[basis + o] = result;
                    }
                }
            }
        }

        Tensor::new(output_shape, output_strides, o_size, values)
    }

    pub fn _conv_transpose(
        &self,
        kernel: &Tensor<DType>,
        _dilations: &Vec<usize>,
        group: usize,
        _pads: &Vec<usize>,
        _strides: &Vec<usize>,
    ) -> Tensor<DType> {
        let N = self.get_dim_size(0);
        let C = self.get_dim_size(1);
        let D = self.get_sh();
        let W = kernel.get_sh();
        let M = kernel.get_dim_size(0);
        let CG = kernel.get_dim_size(1);

        let kernel_size = get_size_from(W, 2);

        let data_rank = self.rank() - 2;

        let R = conv_transpose_output_size(D, W, _pads, _dilations, _strides, 2);
        let output_size = get_size(&R);

        let mut output_shape = vec![0; data_rank + 2];
        output_shape[0] = N;
        output_shape[1] = M;
        for i in 0..data_rank {
            output_shape[i + 2] = R[i];
        }

        let output_strides = compute_strides(&output_shape);
        let o_size = get_size(&output_shape);
        let mut values = vec![zero(); o_size];

        // Iterate over all batches
        for n in 0..N {
            // Iterate over all output channels
            for m in 0..M {
                let basis = output_strides[0] * n + output_strides[1] * m;

                for cg in 0..CG {
                    let c = (m * CG + cg) % C;

                    let mut output_indices = vec![0; data_rank + 2];
                    output_indices[0] = n;
                    output_indices[1] = m;

                    for o_ix in 0..output_size {
                        let mut result = values[basis + o_ix];

                        let mut kernel_indices = vec![0; data_rank + 2];
                        kernel_indices[0] = m;
                        kernel_indices[1] = cg;
                        let kernel_base =
                            m * kernel.get_strides_at(0) + cg * kernel.get_strides_at(1);

                        for kernel_ix in 0..kernel_size {
                            let mut input_ix = vec![0; data_rank + 2];
                            input_ix[0] = n;
                            input_ix[1] = c;

                            let mut skip = false;
                            for axis in 0..data_rank {
                                let trans_kernel_ix =
                                    kernel.get_dim_size(axis + 2) - kernel_indices[axis + 2] - 1;

                                let mut ix = (output_indices[axis + 2] as i32)
                                    - (_pads[axis] as i32)
                                    + (trans_kernel_ix as i32 * _dilations[axis] as i32);

                                let res = (ix as i32) % (_strides[axis] as i32);

                                if res != 0 {
                                    skip = true;
                                    break;
                                }

                                ix = ix / (_strides[axis] as i32);

                                if ix < 0 || ix >= D[axis + 2] as i32 {
                                    skip = true;
                                    break;
                                }

                                input_ix[axis + 2] = ix as usize;
                            }

                            if !skip {
                                let w_i = kernel.get_ix(kernel_base + kernel_ix);
                                let x_i = self.get(&input_ix);
                                result = result + w_i * x_i;
                            }

                            increment_index(&mut kernel_indices, kernel.get_sh());
                        }

                        values[basis + o_ix] = result;

                        increment_index(&mut output_indices, &output_shape);
                    }
                }
            }
        }

        Tensor::new(output_shape, output_strides, o_size, values)
    }

    pub fn _average_pool(
        &self,
        kernel_shape: &Vec<usize>,
        pads: &Vec<usize>,
        strides: &Vec<usize>,
        include_pad: bool,
    ) -> Tensor<DType> {
        let N = self.get_dim_size(0);
        let C = self.get_dim_size(1);
        let D = self.get_sh();

        let kernel_size = get_size(&kernel_shape);

        let data_rank = self.rank() - 2;

        let R = conv_output_size(D, kernel_shape, pads, &vec![1; data_rank], strides, 0);
        let output_size = get_size(&R);

        let mut output_shape = vec![0; data_rank + 2];
        output_shape[0] = N;
        output_shape[1] = C;
        for i in 0..data_rank {
            output_shape[i + 2] = R[i];
        }

        let output_strides = compute_strides(&output_shape);
        let o_size = get_size(&output_shape);
        let mut values = vec![zero(); o_size];

        // Iterate over all batches
        for n in 0..N {
            // Iterate over all output channels
            for c in 0..C {
                let basis = output_strides[0] * n + output_strides[1] * c;

                let mut output_indices = vec![0; data_rank + 2];
                output_indices[0] = n;
                output_indices[1] = c;

                for o_ix in 0..output_size {
                    let mut result = zero();
                    let mut count = 0;

                    let mut kernel_indices = vec![0; data_rank];

                    for _ in 0..kernel_size {
                        let mut input_ix = vec![0; data_rank + 2];
                        input_ix[0] = n;
                        input_ix[1] = c;

                        let mut skip = false;
                        for axis in 0..data_rank {
                            let ix = (output_indices[axis + 2] * strides[axis]) as i32
                                + (kernel_indices[axis]) as i32
                                - (pads[axis] as i32);

                            if ix < 0 || ix >= D[axis + 2] as i32 {
                                skip = true;
                                break;
                            }

                            input_ix[axis + 2] = ix as usize;
                        }

                        if !skip {
                            let x_i = self.get(&input_ix);
                            result = result + x_i;
                        }

                        if !skip || include_pad {
                            count += 1;
                        }

                        increment_index(&mut kernel_indices, kernel_shape);
                    }

                    match DType::from_u32(count) {
                        Some(co) => result = result / co,
                        None => panic!("DType can not represent kernel size in average pool"),
                    }

                    values[basis + o_ix] = result;

                    increment_index(&mut output_indices, &output_shape);
                }
            }
        }

        Tensor::new(output_shape, output_strides, o_size, values)
    }

    pub fn _pad(&self, pads: &Vec<usize>, mode: i32, value: DType) -> Tensor<DType> {
        let rank = self.rank();

        let mut output_shape = vec![0; rank];
        for i in 0..rank {
            output_shape[i] = self.get_dim_size(i) + pads[i] + pads[i + rank];
        }
        let output_strides = compute_strides(&output_shape);
        let output_size = get_size(&output_shape);

        let mut values = vec![zero(); output_size];

        let mut ix = vec![0; rank];
        let mut input_ix = vec![0; rank];
        for i in 0..output_size {
            let mut use_const = false;
            for j in 0..rank {
                if ix[j] < pads[j] {
                    if mode == 0 {
                        use_const = true;
                        break;
                    } else if mode == 1 {
                        input_ix[j] = pads[j] - ix[j];
                    } else {
                        input_ix[j] = 0;
                    }
                } else {
                    input_ix[j] = ix[j] - pads[j];
                }
                if input_ix[j] >= self.get_dim_size(j) {
                    if mode == 0 {
                        use_const = true;
                        break;
                    } else if mode == 1 {
                        input_ix[j] = 2 * self.get_dim_size(j) - input_ix[j] - 2;
                    } else {
                        input_ix[j] = self.get_dim_size(j) - 1;
                    }
                }
            }

            if use_const {
                values[i] = value;
            } else {
                values[i] = self.get(&input_ix);
            }

            increment_index(&mut ix, &output_shape);
        }

        Tensor::new(output_shape, output_strides, output_size, values)
    }

    pub fn _upsample(&self, scales: &Vec<f32>) -> Tensor<DType> {
        let rank = self.rank();
        let mut result_shape = vec![0; rank];
        let mut ax_ix = 0;
        for i in 0..rank {
            result_shape[i] = ((self.get_dim_size(i) as f32) * scales[i]).floor() as usize;
        }

        let result_strides = compute_strides(&result_shape);
        let result_size = get_size(&result_shape);

        let mut values = vec![zero(); result_size];

        let mut out_ix = vec![0; rank];
        let mut in_ix = vec![0; rank];

        for i in 0..result_size {
            for j in 0..rank {
                in_ix[j] = ((out_ix[j] as f32) / scales[j]).floor() as usize;
            }

            values[i] = self.get(&in_ix);

            increment_index(&mut out_ix, &result_shape);
        }

        Tensor::new(result_shape, result_strides, result_size, values)
    }
}

impl<DType> Tensor<DType>
where
    DType: Copy,
    DType: Num,
    DType: FromPrimitive,
    DType: PartialOrd,
{
    pub fn conv(
        &self,
        kernel: &Tensor<DType>,
        dilations: Uint32Array,
        group: u32,
        pads: Uint32Array,
        strides: Uint32Array,
        activation: u32,
    ) -> Tensor<DType> {
        let mut _dilations: Vec<usize> = vec![0; dilations.length() as usize];
        let mut _pads: Vec<usize> = vec![0; pads.length() as usize];
        let mut _strides: Vec<usize> = vec![0; strides.length() as usize];
        for i in 0..dilations.length() {
            _dilations[i as usize] = dilations.get_index(i) as usize;
            _pads[i as usize] = pads.get_index(i) as usize;
            _pads[(i + dilations.length()) as usize] =
                pads.get_index(i + dilations.length()) as usize;
            _strides[i as usize] = strides.get_index(i) as usize;
        }

        return self._conv(
            kernel,
            None,
            &_dilations,
            group as usize,
            &_pads,
            &_strides,
            activation,
        );
    }

    pub fn conv_with_bias(
        &self,
        kernel: &Tensor<DType>,
        bias: &Tensor<DType>,
        dilations: Uint32Array,
        group: u32,
        pads: Uint32Array,
        strides: Uint32Array,
        activation: u32,
    ) -> Tensor<DType> {
        let mut _dilations: Vec<usize> = vec![0; dilations.length() as usize];
        let mut _pads: Vec<usize> = vec![0; pads.length() as usize];
        let mut _strides: Vec<usize> = vec![0; strides.length() as usize];
        for i in 0..dilations.length() {
            _dilations[i as usize] = dilations.get_index(i) as usize;
            _pads[i as usize] = pads.get_index(i) as usize;
            _pads[(i + dilations.length()) as usize] =
                pads.get_index(i + dilations.length()) as usize;
            _strides[i as usize] = strides.get_index(i) as usize;
        }

        return self._conv(
            kernel,
            Some(bias),
            &_dilations,
            group as usize,
            &_pads,
            &_strides,
            activation,
        );
    }

    pub fn conv_transpose(
        &self,
        kernel: &Tensor<DType>,
        dilations: Uint32Array,
        group: u32,
        pads: Uint32Array,
        strides: Uint32Array,
    ) -> Tensor<DType> {
        let mut _dilations: Vec<usize> = vec![0; dilations.length() as usize];
        let mut _pads: Vec<usize> = vec![0; pads.length() as usize];
        let mut _strides: Vec<usize> = vec![0; strides.length() as usize];
        for i in 0..dilations.length() {
            _dilations[i as usize] = dilations.get_index(i) as usize;
            _pads[i as usize] = pads.get_index(i) as usize;
            _pads[(i + dilations.length()) as usize] =
                pads.get_index(i + dilations.length()) as usize;
            _strides[i as usize] = strides.get_index(i) as usize;
        }

        return self._conv_transpose(kernel, &_dilations, group as usize, &_pads, &_strides);
    }

    pub fn average_pool(
        &self,
        kernel_shape: Uint32Array,
        pads: Uint32Array,
        strides: Uint32Array,
        include_pad: bool,
    ) -> Tensor<DType> {
        let mut _kernel_shape: Vec<usize> = vec![0; kernel_shape.length() as usize];
        let mut _pads: Vec<usize> = vec![0; pads.length() as usize];
        let mut _strides: Vec<usize> = vec![0; strides.length() as usize];
        for i in 0..kernel_shape.length() {
            _kernel_shape[i as usize] = kernel_shape.get_index(i) as usize;
            _pads[i as usize] = pads.get_index(i) as usize;
            _pads[(i + kernel_shape.length()) as usize] =
                pads.get_index(i + kernel_shape.length()) as usize;
            _strides[i as usize] = strides.get_index(i) as usize;
        }

        return self._average_pool(&_kernel_shape, &_pads, &_strides, include_pad);
    }

    // Mode: 0 == constant, 1 == reflect, 2 == edge
    pub fn pad(&self, pads: Uint32Array, mode: i32, value: DType) -> Tensor<DType> {
        let mut _pads: Vec<usize> = vec![0; pads.length() as usize];
        for i in 0..pads.length() {
            _pads[i as usize] = pads.get_index(i) as usize;
        }
        return self._pad(&_pads, mode, value);
    }

    pub fn upsample(&self, scales: Float32Array) -> Tensor<DType> {
        let mut _scales: Vec<f32> = vec![0.0; scales.length() as usize];
        for i in 0..scales.length() {
            _scales[i as usize] = scales.get_index(i) as f32;
        }

        return self._upsample(&_scales);
    }
}

impl<DType> Tensor<DType>
where
    DType: Clone,
    DType: Num,
    DType: PartialOrd,
    DType: Float,
{
    pub fn normalize(
        &self,
        mean: &Tensor<DType>,
        variance: &Tensor<DType>,
        epsilon: DType,
        scale: &Tensor<DType>,
        bias: &Tensor<DType>,
    ) -> Tensor<DType> {
        let mut result_shape = vec![0; self.rank()];
        for i in 0..self.rank() {
            result_shape[i] = self.get_dim_size(i);
        }
        let result_strides = compute_strides(&result_shape);

        let mut values = vec![zero(); self.size];

        let mut out_ix = vec![0; self.rank()];
        for i in 0..self.size {
            let mut res = self.get_ix(i) - mean.get(&out_ix);
            res = res / (variance.get(&out_ix) + epsilon).sqrt();
            res = res * scale.get(&out_ix) + bias.get(&out_ix);

            values[i] = res;

            increment_index(&mut out_ix, &result_shape);
        }

        Tensor::new(result_shape, result_strides, self.size, values)
    }
}
