use js_sys::Int32Array;
use crate::utils::conv_output_size;
use std::cmp::Ordering;
use std::ops::{Add, Sub};
use wasm_bindgen::prelude::*;

use js_sys::{Uint32Array, Float32Array};

use crate::shape::*;
use crate::utils::*;

use std::cmp;

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    pub size: usize,
    values: Vec<f32>
}

impl Tensor {
    pub fn new(shape: &Vec<usize>, values: &Vec<f32>) -> Tensor {
        let strides = compute_strides(shape);
        let size = get_size(shape);

        Tensor {
            shape: shape.to_vec(),
            strides,
            size,
            values: values.to_vec() // TODO: There must be a different way to do this
        }
    }

    pub fn constant(shape: &Vec<usize>, value: f32) -> Tensor {
        let strides = compute_strides(shape);
        let size = get_size(shape);

        let values = vec![value; size];

        Tensor {
            shape: shape.to_vec(),
            strides,
            size,
            values
        }
    }

    pub fn get(& self, index: &Vec<usize>) -> f32 {
        let pos = index_to_pos(index, &self.strides);
        return self.values[pos];
    }

    pub fn set(&mut self, index: &Vec<usize>, value: f32) {
        let pos = index_to_pos(index, &self.strides);
        self.values[pos] = value;
    }

    pub fn get_values(&self) -> &Vec<f32> {
        return &self.values;
    }

    pub fn compare(&self, other: &Self, delta: f32) -> bool {
        if !compare_shapes(&self.shape, &other.shape) {
            return false;
        }

        for i in 0..self.size {
            if (self.values[i] - other.get_values()[i]).abs() > delta {
                return false;
            }
        }

        return true;
    }

    #[inline]
    fn unary_op<F>(&self, op: F) -> Tensor where F: Fn(f32) -> f32 {
        let mut values: Vec<f32> = vec![0.0; self.size];
        for i in 0..self.size {
            values[i] = op(self.values[i]);
        }

        Tensor {
            values,
            shape: self.shape.to_vec(),
            size: self.size,
            strides: self.strides.to_vec()
        }
    }

    #[inline]
    fn binary_op<F>(&self, other: &Tensor, op: F) -> Tensor where F: Fn(f32, f32) -> f32 {
        let mut result_shape = vec![0; self.shape.len()];
        for i in 0..self.shape.len() {
            result_shape[i] = cmp::max(self.shape[i], other.shape[i]);
        }
        let result_size = get_size(&result_shape);
        let result_strides = compute_strides(&result_shape);

        let mut values: Vec<f32> = vec![0.0; result_size];

        let mut ix = vec![0; self.shape.len()];

        for i in 0..result_size {
            values[i] = op(self.get(&ix), other.get(&ix));

            increment_index(&mut ix, &result_shape);
        }

        Tensor {
            values,
            shape: result_shape,
            size: result_size,
            strides: result_strides
        }
    }

    pub fn axes_continuous(&self, axes: &Vec<usize>) -> bool {
        let mut last_ax = axes[0];
        for i in 1..axes.len() {
            if axes[i] > last_ax + 1 {
                return false;
            }
            last_ax = axes[i];
        }
        return true;
    }

    #[inline]
    pub fn pool_continuous<F,F2,F3>(&self, axes: &Vec<usize>, keep_dims: bool,
                                 op: F, postprocess: bool, post: F2, init: bool, init_func: F3) -> Tensor where F: Fn(f32,f32) -> f32, F2: Fn(f32) -> f32, F3: Fn(f32) -> f32 {
        let mut result_rank = self.shape.len() - axes.len() as usize;

        if keep_dims {
            result_rank = self.shape.len()
        }

        let mut result_shape = vec![0; result_rank];
        let mut result_size = 1;
        let mut sum_size = 1;
        let mut axes_ix = 0;
        let mut result_ix = 0;
        for i in 0..self.shape.len() {
            if axes_ix < axes.len() && axes[axes_ix] as usize == i {
                axes_ix += 1;
                sum_size *= self.shape[i];
                if keep_dims {
                    result_shape[result_ix] = 1;
                    result_ix += 1;
                }
            } else {
                result_shape[result_ix] = self.shape[i];
                result_ix += 1;
                result_size *= self.shape[i];
            }
        }
        let result_strides = compute_strides(&result_shape);
        let mut values = vec![0.0; result_size];


        let step_size = self.strides[axes[axes.len() - 1]];
        let cont_size = step_size;


        let input_ix_step_size = if axes[0] > 0 { self.strides[axes[0]-1] } else { self.size };
        let num_input_steps = self.size/input_ix_step_size;

        let mut input_start_ix;
        let mut output_ix;
        for i in 0..num_input_steps {
            input_start_ix = i*input_ix_step_size;
            output_ix = i*cont_size;

            for j in 0..cont_size {
                let mut res = self.values[input_start_ix + j];
                if init {
                    res = init_func(res);
                }
                for k in 1..sum_size {
                    res = op(self.values[input_start_ix + j + k*step_size], res);
                }
                if postprocess {
                    res = post(res);
                }
                values[output_ix+j] = res;
            }
        }

        Tensor {
            values,
            shape: result_shape,
            size: result_size,
            strides: result_strides
        }
    }

    #[inline]
    pub fn _pool<F, F2, F3>(&self, axes: &Vec<usize>, keep_dims: bool,
                        op: F, postprocess: bool, post: F2,
                        init: bool, init_func: F3) -> Tensor where F: Fn(f32,f32) -> f32, F2: Fn(f32) -> f32, F3: Fn(f32) -> f32 {
        let mut result_rank = self.shape.len() - axes.len() as usize;

        if keep_dims {
            result_rank = self.shape.len();
        }

        if result_rank == 0 {
            let mut value = self.values[0];
            if init {
                value = init_func(value);
            }
            for i in 1..self.size {
                value = op(self.values[i], value);
            }

            if postprocess {
                value = post(value);
            }

            return Tensor {
                values: vec![value],
                shape: if keep_dims { vec![1;result_rank] } else { vec![1;1] },
                size: 1,
                strides: vec![1]
            }
        }

        if self.axes_continuous(axes) {
            return self.pool_continuous(axes, keep_dims, op, postprocess, post, init, init_func);
        }

        let mut result_shape = vec![0; result_rank];
        let mut result_size = 1;
        let mut axes_ix = 0;
        let mut result_ix = 0;
        let mut res_ix_map = vec![0; result_rank];
        for i in 0..self.shape.len() {
            if axes_ix < axes.len() && axes[axes_ix] as usize == i {
                axes_ix += 1;
                if keep_dims {
                    result_shape[result_ix] = 1;
                    res_ix_map[result_ix] = i;
                    result_ix += 1;
                }
            } else {
                result_shape[result_ix] = self.shape[i];
                res_ix_map[result_ix] = i;
                result_ix += 1;
                result_size *= self.shape[i];
            }
        }
        let result_strides = compute_strides(&result_shape);
        let mut values = vec![0.0; result_size];
        let mut initialized = vec![false; result_size];

        let mut input_index = vec![0; self.shape.len()];
        for i in 0..self.size {
            let mut res_ix = 0;
            for j in 0..result_rank {
                res_ix += result_strides[j] * input_index[res_ix_map[j]];
            }
            if !initialized[res_ix] {
                values[res_ix] = self.values[i];
                if init {
                    values[res_ix] = init_func(values[res_ix]);
                }
                initialized[res_ix] = true;
            } else {
                values[res_ix] = op(self.values[i], values[res_ix]);
            }

            increment_index(&mut input_index, &self.shape);
        }

        if postprocess {
            for i in 0..result_size {
                values[i] = post(values[i]);
            }
        }

        Tensor {
            values,
            shape: result_shape,
            size: result_size,
            strides: result_strides
        }
    }

    pub fn _sum(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor {
        return self._pool(axes, keep_dims, |x: f32, y: f32| x+y, false, |x: f32| x, false, |x: f32| x)
    }

    pub fn _sum_square(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor {
        return self._pool(axes, keep_dims, |x: f32, y: f32| (x*x)+y, false, |x: f32| x, true, |x: f32| x*x)
    }

    pub fn _product(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor {
        return self._pool(axes, keep_dims, |x: f32, y: f32| x*y, false, |x: f32| x, false, |x: f32| x)
    }

    pub fn _max(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor {
        return self._pool(axes, keep_dims, |x: f32, y: f32| x.max(y), false, |x: f32| x, false, |x: f32| x)
    }

    pub fn _min(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor {
        return self._pool(axes, keep_dims, |x: f32, y: f32| x.min(y), false, |x: f32| x, false, |x: f32| x)
    }

    pub fn _reduce_mean(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor {
        let mut pool_size = 1;
        for i in 0..axes.len() {
            pool_size *= self.shape[axes[i]];
        }
        let pool_size_f = pool_size as f32;

        return self._pool(axes, keep_dims, |x: f32, y: f32| x + y, true, |x: f32| x/pool_size_f, false, |x: f32| x)
    }

    pub fn _reduce_mean_square(&self, axes: &Vec<usize>, keep_dims: bool) -> Tensor {
        let mut pool_size = 1;
        for i in 0..axes.len() {
            pool_size *= self.shape[axes[i]];
        }
        let pool_size_f = pool_size as f32;

        return self._pool(axes, keep_dims, |x: f32, y: f32| (x*x) + y, true, |x: f32| x/pool_size_f, true, |x: f32| x*x)
    }

    pub fn _conv(&self,
                 kernel: &Tensor,
                 bias: Option<&Tensor>,
                 _dilations: &Vec<usize>,
                 group: usize,
                 _pads: &Vec<usize>,
                 _strides: &Vec<usize>
                 ) -> Tensor {
        let N = self.shape[0];
        let C = self.shape[1];
        let D = &self.shape;
        let W = &kernel.shape;
        let M = kernel.shape[0];
        let CG = kernel.shape[1];

        let kernel_size = get_size_from(W, 2);

        let data_rank = self.shape.len() - 2;

        let R = conv_output_size(D, W, _pads, _dilations, _strides, 2);
        let output_size = get_size(&R);

        let mut output_shape = vec![0; data_rank + 2];
        output_shape[0] = N;
        output_shape[1] = M;
        for i in 0..data_rank {
            output_shape[i+2] = R[i];
        }
        
        let output_strides = compute_strides(&output_shape);
        let o_size = get_size(&output_shape);
        let mut values = vec![0.0; o_size];

        // Iterate over all batches
        for n in 0..N {
            // Iterate over all output channels
            for m in 0..M {
                let basis = output_strides[0] * n + output_strides[1] * m;

                match bias {
                    None => {},
                    Some(bi) => {
                        let b = bi.values[m];

                        for o in 0..output_size {
                            values[basis + o] = b;
                        }
                    }
                }


                for cg in 0..CG {
                    let c = (m * CG + cg)%C;

                    let mut output_indices = vec![0; data_rank + 2];
                    output_indices[0] = n;
                    output_indices[1] = m;

                    for o_ix in 0..output_size {
                        let mut result = values[basis + o_ix];

                        let mut kernel_indices = vec![0; data_rank + 2];
                        kernel_indices[0] = m;
                        kernel_indices[1] = cg;
                        let kernel_base = m*kernel.strides[0] + cg*kernel.strides[1];

                        for kernel_ix in 0..kernel_size {
                            let mut input_ix = vec![0; data_rank + 2];
                            input_ix[0] = n;
                            input_ix[1] = c;

                            let mut skip = false;
                            for axis in 0..data_rank {
                                let ix = (output_indices[axis + 2] * _strides[axis]) as i32 + (kernel_indices[axis + 2] * _dilations[axis]) as i32 - (_pads[axis] as i32);

                                if ix < 0 || ix >= D[axis + 2] as i32 {
                                    skip = true;
                                    break;
                                }

                                input_ix[axis + 2] = ix as usize;
                            }

                            if !skip {
                                let w_i = kernel.values[kernel_base + kernel_ix];
                                let x_i = self.get(&input_ix);
                                result += w_i * x_i;
                            }

                            increment_index(&mut kernel_indices, &kernel.shape);
                        }

                        values[basis + o_ix] = result;

                        increment_index(&mut output_indices, &output_shape);
                    }
                }
            }
        }

        Tensor {
            values,
            size: o_size,
            shape: output_shape,
            strides: output_strides
        }
    }

    pub fn _average_pool(&self,
                         kernel_shape: &Vec<usize>,
                         pads: &Vec<usize>,
                         strides: &Vec<usize>,
                         include_pad: bool) -> Tensor {
        let N = self.shape[0];
        let C = self.shape[1];
        let D = &self.shape;

        let kernel_size = get_size(&kernel_shape);

        let data_rank = self.shape.len() - 2;

        let R = conv_output_size(D, kernel_shape, pads, &vec![1; data_rank], strides, 0);
        let output_size = get_size(&R);

        let mut output_shape = vec![0; data_rank + 2];
        output_shape[0] = N;
        output_shape[1] = C;
        for i in 0..data_rank {
            output_shape[i+2] = R[i];
        }

        let output_strides = compute_strides(&output_shape);
        let o_size = get_size(&output_shape);
        let mut values = vec![0.0; o_size];

        // Iterate over all batches
        for n in 0..N {
            // Iterate over all output channels
            for c in 0..C {
                let basis = output_strides[0] * n + output_strides[1] * c;

                let mut output_indices = vec![0; data_rank + 2];
                output_indices[0] = n;
                output_indices[1] = c;

                for o_ix in 0..output_size {
                    let mut result = 0.0;
                    let mut count = 0;

                    let mut kernel_indices = vec![0; data_rank];

                    for _ in 0..kernel_size {
                        let mut input_ix = vec![0; data_rank + 2];
                        input_ix[0] = n;
                        input_ix[1] = c;

                        let mut skip = false;
                        for axis in 0..data_rank {
                            let ix = (output_indices[axis + 2] * strides[axis]) as i32 + (kernel_indices[axis]) as i32 - (pads[axis] as i32);

                            if ix < 0 || ix >= D[axis + 2] as i32 {
                                skip = true;
                                break;
                            }

                            input_ix[axis + 2] = ix as usize;
                        }

                        if !skip {
                            let x_i = self.get(&input_ix);
                            result += x_i;
                        }

                        if !skip || include_pad {
                            count += 1;
                        }

                        increment_index(&mut kernel_indices, kernel_shape);
                    }

                    result = result / (count as f32);

                    values[basis + o_ix] = result;

                    increment_index(&mut output_indices, &output_shape);
                }
            }
        }

        Tensor {
            values,
            size: o_size,
            shape: output_shape,
            strides: output_strides
        }
    }

    pub fn _reshape(&self, shape: &Vec<usize>) -> Tensor {
        let strides = compute_strides(shape);

        let mut values = vec![0.0; self.size];
        for i in 0..self.size {
            values[i] = self.values[i];
        }

        Tensor {
            values,
            size: self.size,
            shape: shape.to_vec(),
            strides: strides
        }
    }

    pub fn _gemm(&self, b: &Tensor, a_transpose: bool, b_transpose: bool, alpha: f32, c: Option<&Tensor>, beta: f32) -> Tensor {
        let rank = self.shape.len();

        let M = if a_transpose { self.shape[rank - 1] } else { self.shape[rank - 2]};
        let N = if a_transpose { self.shape[rank - 2] } else { self.shape[rank - 1]};
        let O = if b_transpose { b.shape[rank - 2] } else { b.shape[rank - 1]};

        let a_batch_mult = M*N;
        let b_batch_mult = N*O;
        let y_batch_mult = M*O;

        let a_n_mult = if a_transpose { M } else { 1 };
        let a_m_mult = if a_transpose { 1 } else { N };
        let b_n_mult = if b_transpose { 1 } else { O };
        let b_o_mult = if b_transpose { N } else { 1 };

        let mut c_m_mult = 0;
        let mut c_o_mult = 0;
        let mut c_batch_mult = 0;
        match c {
            None => {},
            Some(c_) => {
                let c_batch_size = get_size_from_to(&c_.shape, 0, rank-2);

                c_m_mult = c_.strides[rank - 2];
                c_o_mult = c_.strides[rank - 1];
                if c_batch_size > 1 {
                    c_batch_mult = c_.shape[rank-2]*c_.shape[rank-1];

                    if c_batch_mult == 1 {
                        c_batch_mult = 0;
                    }
                } else {
                    c_batch_mult = 0;
                }
            }
        }

        let mut batch_size = get_size_from_to(&self.shape, 0, rank-2);
        if batch_size == 0 {
            batch_size = 1;
        }
        let mut result_shape = vec![0; rank];
        result_shape[rank - 1] = O;
        result_shape[rank - 2] = M;
        for i in 0..(rank-2) {
            result_shape[i] = self.shape[i];
        }
        let result_size = get_size(&result_shape);
        let result_strides = compute_strides(&result_shape);

        let mut values = vec![0.0; result_size];

        for i in 0..batch_size {
            let a_base = i*a_batch_mult;
            let b_base = i*b_batch_mult;
            let y_base = i*y_batch_mult;
            let c_base = i*c_batch_mult;

            for m in 0..M {
                for o in 0..O {
                    let mut result = 0.0;

                    for n in 0..N {
                        result += self.values[a_base + m*a_m_mult + n*a_n_mult] * b.values[b_base + n*b_n_mult + o*b_o_mult];
                    }

                    result = alpha*result;
                    match c {
                        None => {},
                        Some(c_) => {
                            let ix = c_base + m*c_m_mult + o*c_o_mult;
                            result += beta*c_.values[ix];
                        }
                    }
                    values[y_base + m*O + o] = result;
                }
            }
        }

        Tensor {
            shape: result_shape,
            size: result_size,
            values,
            strides: result_strides
        }
    }

    pub fn _transpose(&self, permutation: &Vec<usize>) -> Tensor {
        let rank = self.shape.len();

        let mut output_shape = vec![0; rank];
        let mut reverse_perm = vec![0; rank];
        for i in 0..rank {
            output_shape[i] = self.shape[permutation[i]];
            reverse_perm[permutation[i]] = i;
        }

        let output_strides = compute_strides(&output_shape);
        let mut mapped_strides = vec![0; rank];
        for i in 0..rank {
            mapped_strides[i] = output_strides[reverse_perm[i]];
        }

        let mut values = vec![0.0; self.size];

        let mut index = vec![0; rank];
        for i in 0..self.size {
            let mut out_ix = 0;
            for j in 0..rank {
                out_ix += index[j]*mapped_strides[j];
            }

            values[out_ix] = self.values[i];

            increment_index(&mut index, &self.shape);
        }

        Tensor {
            shape: output_shape,
            strides: output_strides,
            size: self.size,
            values
        }
    }

    pub fn _repeat(&self, repeats: &Vec<usize>) -> Tensor {
        let rank = self.shape.len();

        let mut output_shape = vec![0; rank];
        for i in 0..rank {
            output_shape[i] = self.shape[i] * repeats[i];
        }

        let output_strides = compute_strides(&output_shape);
        let output_size = get_size(&output_shape);

        let mut values = vec![0.0; output_size];

        let mut index = vec![0; rank];
        let mut in_ix = vec![0; rank];
        for i in 0..output_size {
            for j in 0..rank {
                in_ix[j] = index[j] % self.shape[j];
            }

            values[i] = self.get(&in_ix);

            increment_index(&mut index, &output_shape);
        }

        Tensor {
            shape: output_shape,
            strides: output_strides,
            size: output_size,
            values
        }
    }

    pub fn _expand(&self, shape: &Vec<usize>) -> Tensor {
        let mut output_shape = vec![0; shape.len()];
        for i in 0..shape.len() {
            output_shape[i] = shape[i]
        }
        let output_strides = compute_strides(&output_shape);
        let output_size = get_size(&output_shape);

        let mut values = vec![0.0; output_size];

        let mut ix = vec![0; shape.len()];
        for i in 0..output_size {
            values[i] = self.get(&ix);

            increment_index(&mut ix, &output_shape);
        }

        Tensor {
            shape: output_shape,
            strides: output_strides,
            size: output_size,
            values
        }
    }

    pub fn _pad(&self, pads: &Vec<usize>, mode: i32, value: f32) -> Tensor {
        let rank = self.shape.len();

        let mut output_shape = vec![0; rank];
        for i in 0..rank {
            output_shape[i] = self.shape[i] + pads[i] + pads[i+rank];
        }
        let output_strides = compute_strides(&output_shape);
        let output_size = get_size(&output_shape);

        let mut values = vec![0.0; output_size];

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
                if input_ix[j] >= self.shape[j] {
                    if mode == 0 {
                        use_const = true;
                        break;
                    } else if mode == 1 {
                        input_ix[j] = 2*self.shape[j] - input_ix[j] - 2;
                    } else {
                        input_ix[j] = self.shape[j] - 1;
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

        Tensor {
            shape: output_shape,
            strides: output_strides,
            size: output_size,
            values
        }
    }
}

#[wasm_bindgen]
impl Tensor {
    pub fn create(shape: Uint32Array, values: Float32Array) -> Tensor {
        let mut _shape: Vec<usize> = vec![0; shape.length() as usize];
        for i in 0.._shape.len() {
            _shape[i] = shape.get_index(i as u32) as usize;
        }

        let strides = compute_strides(&_shape);
        let size = get_size(&_shape);

        let mut _values: Vec<f32> = vec![0.; values.length() as usize];
        for i in 0.._values.len() {
            _values[i] = values.get_index(i as u32);
        }

        Tensor {
            shape: _shape,
            strides,
            size,
            values: _values
        }
    }

    pub fn create_constant(shape: Uint32Array, value: f32) -> Tensor {
        let mut _shape: Vec<usize> = vec![0; shape.length() as usize];
        for i in 0.._shape.len() {
            _shape[i] = shape.get_index(i as u32) as usize;
        }

        let strides = compute_strides(&_shape);
        let size = get_size(&_shape);

        let values = vec![value; size];

        Tensor {
            shape: _shape,
            strides,
            size,
            values
        }
    }

    pub fn exp(&self) -> Tensor {
        self.unary_op(|x: f32| x.exp())
    }

    pub fn log(&self) -> Tensor {
        self.unary_op(|x: f32| x.ln())
    }

    pub fn sqrt(&self) -> Tensor {
        self.unary_op(|x: f32| x.sqrt())
    }

    pub fn abs(&self) -> Tensor {
        self.unary_op(|x: f32| x.abs())
    }

    pub fn addition(&self, other: &Tensor) -> Tensor {
        return self.binary_op(other, |x: f32, y: f32| x + y);
    }

    pub fn subtraction(&self, other: &Tensor) -> Tensor {
        return self.binary_op(other, |x: f32, y: f32| x - y);
    }

    pub fn multiply(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, |x: f32, y: f32| x * y)
    }

    pub fn divide(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, |x: f32, y: f32| x / y)
    }

    pub fn power(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, |x: f32, y: f32| x.powf(y))
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let m = self.shape[0];
        let n = self.shape[1];
        let o = other.shape[1];

        let mut values = vec![0.; m*o];
        for i in 0..m {
            for k in 0..o {
                let mut res = 0.;
                for j in 0..n {
                    res += self.values[i*n + j] * other.values[j*o + k];
                }
                values[i*o + k] = res;
            }
        }

        Tensor {
            values,
            shape: vec![m,o],
            size: m*o,
            strides: vec![o, 1]
        }
    }

    pub fn gemm(&self, other: &Tensor, a_transpose: bool, b_transpose: bool, alpha: f32) -> Tensor {
        return self._gemm(other, a_transpose, b_transpose, alpha, None, 1.0);
    }

    pub fn gemm_with_c(&self, other: &Tensor, a_transpose: bool, b_transpose: bool, alpha: f32, c: &Tensor, beta: f32) -> Tensor {
        return self._gemm(other, a_transpose, b_transpose, alpha, Some(c), beta);
    }

    pub fn get_vals(&self) -> Float32Array {
        let arr = Float32Array::new_with_length(self.values.len() as u32);

        for i in 0..self.values.len() {
            arr.set_index(i as u32, self.values[i]);
        }

        return arr;
    }

    pub fn get_shape(&self) -> Uint32Array {
        let arr = Uint32Array::new_with_length(self.shape.len() as u32);

        for i in 0..self.shape.len() {
            arr.set_index(i as u32, self.shape[i] as u32);
        }

        return arr;
    }

    pub fn sum(&self, axes: Uint32Array, keep_dims: bool) -> Tensor {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._sum(&ax, keep_dims);
    }

    pub fn sum_square(&self, axes: Uint32Array, keep_dims: bool) -> Tensor {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._sum_square(&ax, keep_dims);
    }

    pub fn product(&self, axes: Uint32Array, keep_dims: bool) -> Tensor {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._product(&ax, keep_dims);
    }

    pub fn max(&self, axes: Uint32Array, keep_dims: bool) -> Tensor {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._max(&ax, keep_dims);
    }

    pub fn min(&self, axes: Uint32Array, keep_dims: bool) -> Tensor {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._min(&ax, keep_dims);
    }

    pub fn reduce_mean(&self, axes: Uint32Array, keep_dims: bool) -> Tensor {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._reduce_mean(&ax, keep_dims);
    }

    pub fn reduce_mean_square(&self, axes: Uint32Array, keep_dims: bool) -> Tensor {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._reduce_mean_square(&ax, keep_dims);
    }

    pub fn conv(&self,
                kernel: &Tensor,
                dilations: Uint32Array,
                group: u32,
                pads: Uint32Array,
                strides: Uint32Array) -> Tensor {
        let mut _dilations: Vec<usize> = vec![0; dilations.length() as usize];
        let mut _pads: Vec<usize> = vec![0; pads.length() as usize];
        let mut _strides: Vec<usize> = vec![0; strides.length() as usize];
        for i in 0..dilations.length() {
            _dilations[i as usize] = dilations.get_index(i) as usize;
            _pads[i as usize] = pads.get_index(i) as usize;
            _pads[(i + dilations.length()) as usize] = pads.get_index(i + dilations.length()) as usize;
            _strides[i as usize] = strides.get_index(i) as usize;
        }

        return self._conv(kernel, None, &_dilations, group as usize, &_pads, &_strides);
    }

    pub fn conv_with_bias(&self,
        kernel: &Tensor,
        bias: &Tensor,
        dilations: Uint32Array,
        group: u32,
        pads: Uint32Array,
        strides: Uint32Array) -> Tensor {
        let mut _dilations: Vec<usize> = vec![0; dilations.length() as usize];
        let mut _pads: Vec<usize> = vec![0; pads.length() as usize];
        let mut _strides: Vec<usize> = vec![0; strides.length() as usize];
        for i in 0..dilations.length() {
            _dilations[i as usize] = dilations.get_index(i) as usize;
            _pads[i as usize] = pads.get_index(i) as usize;
            _pads[(i + dilations.length()) as usize] = pads.get_index(i + dilations.length()) as usize;
            _strides[i as usize] = strides.get_index(i) as usize;
        }

        return self._conv(kernel, Some(bias), &_dilations, group as usize, &_pads, &_strides);
    }

    pub fn average_pool(&self,
                        kernel_shape: Uint32Array,
                        pads: Uint32Array,
                        strides: Uint32Array,
                        include_pad: bool) -> Tensor {
        let mut _kernel_shape: Vec<usize> = vec![0; kernel_shape.length() as usize];
        let mut _pads: Vec<usize> = vec![0; pads.length() as usize];
        let mut _strides: Vec<usize> = vec![0; strides.length() as usize];
        for i in 0..kernel_shape.length() {
            _kernel_shape[i as usize] = kernel_shape.get_index(i) as usize;
            _pads[i as usize] = pads.get_index(i) as usize;
            _pads[(i + kernel_shape.length()) as usize] = pads.get_index(i + kernel_shape.length()) as usize;
            _strides[i as usize] = strides.get_index(i) as usize;
        }

        return self._average_pool(&_kernel_shape, &_pads, &_strides, include_pad);
    }

    pub fn reshape(&self, shape: Uint32Array) -> Tensor {
        let mut sh: Vec<usize> = vec![0; shape.length() as usize];
        for i in 0..shape.length() {
            sh[i as usize] = shape.get_index(i) as usize;
        }
        return self._reshape(&sh);
    }

    pub fn concat(&self, other: &Tensor, axes: u32) -> Tensor {
        let ax = axes as usize;
        let mut output_shape = vec![0; self.shape.len()];
        for i in 0..self.shape.len() {
            output_shape[i] = self.shape[i];
        }
        output_shape[ax] += other.shape[ax];

        let output_strides = compute_strides(&output_shape);
        let output_size = get_size(&output_shape);

        let mut values = vec![0.0; output_size];

        let mut index_x = 0;
        let mut index_y = 0;
        let mut index_out = 0;
        let iter_x_size = output_strides[ax] * self.shape[ax];
        let iter_y_size = output_strides[ax] * other.shape[ax];
        let outer_iters = output_size / (if ax > 0 { output_strides[ax - 1] } else { output_size });
        for _ in 0..outer_iters {
            for _ in 0..iter_x_size {
                values[index_out] = self.values[index_x];
                index_x += 1;
                index_out += 1;
            }

            for _ in 0..iter_y_size {
                values[index_out] = other.values[index_y];
                index_y += 1;
                index_out += 1;
            }
        }

        Tensor {
            shape: output_shape,
            strides: output_strides,
            size: output_size,
            values
        }
    }

    pub fn transpose(&self, permutation: Uint32Array) -> Tensor {
        let mut perm: Vec<usize> = vec![0; permutation.length() as usize];
        for i in 0..permutation.length() {
            perm[i as usize] = permutation.get_index(i) as usize;
        }
        return self._transpose(&perm);
    }

    pub fn repeat(&self, repeats: Uint32Array) -> Tensor {
        let mut _repeats: Vec<usize> = vec![0; repeats.length() as usize];
        for i in 0..repeats.length() {
            _repeats[i as usize] = repeats.get_index(i) as usize;
        }
        return self._repeat(&_repeats);
    }

    pub fn clip(&self, min: f32, max: f32) -> Tensor {
        self.unary_op(|x: f32| x.min(max).max(min))
    }

    pub fn clip_min(&self, min: f32) -> Tensor {
        self.unary_op(|x: f32| x.max(min))
    }

    pub fn clip_max(&self, max: f32) -> Tensor {
        self.unary_op(|x: f32| x.min(max))
    }

    pub fn expand(&self, shape: Uint32Array) -> Tensor {
        let mut _shape: Vec<usize> = vec![0; shape.length() as usize];
        for i in 0..shape.length() {
            _shape[i as usize] = shape.get_index(i) as usize;
        }
        return self._expand(&_shape);
    }

    pub fn copy(&self) -> Tensor {
        let mut _shape: Vec<usize> = vec![0; self.shape.len() as usize];
        let mut _strides: Vec<usize> = vec![0; self.shape.len() as usize];
        for i in 0..self.shape.len() {
            _shape[i] = self.shape[i];
            _strides[i] = self.strides[i];
        }

        let mut values = vec![0.0; self.size];
        for i in 0..self.size {
            values[i] = self.values[i];
        }

        Tensor {
            shape: _shape,
            strides: _strides,
            size: self.size,
            values
        }
    }

    // Mode: 0 == constant, 1 == reflect, 2 == edge
    pub fn pad(&self, pads: Uint32Array, mode: i32, value: f32) -> Tensor {
        let mut _pads: Vec<usize> = vec![0; pads.length() as usize];
        for i in 0..pads.length() {
            _pads[i as usize] = pads.get_index(i) as usize;
        }
        return self._pad(&_pads, mode, value);
    }

    // Mode: 0 == constant, 1 == reflect, 2 == edge
    pub fn gather(&self, axis: i32, indices: Int32Array, indice_shape: Uint32Array) -> Tensor {
        let indice_strides = compute_strides_uint32(&indice_shape);

        let r = self.shape.len();
        let q = indice_shape.length();

        let result_rank = r + (q as usize) - 1;
        let mut result_shape = vec![0; result_rank];
        for i in 0..axis {
            result_shape[i as usize] = self.shape[i as usize];
        }
        for i in 0..q {
            result_shape[(i + (axis as u32)) as usize] = indice_shape.get_index(i) as usize;
        }
        for i in (axis + 1)..(r as i32) {
            result_shape[(i as usize) + (q as usize) - 1] = self.shape[i as usize];
        }

        let result_strides = compute_strides(&result_shape);
        let result_size = get_size(&result_shape);

        let mut values = vec![0.0; result_size];

        let mut out_ix = vec![0; result_rank];
        let mut input_ix = vec![0; self.shape.len()];
        for i in 0..result_size {
            let mut gather_pos = 0;
            for j in 0..q {
                gather_pos += out_ix[j as usize + (axis as usize)] * indice_strides[j as usize];
            }
            let ax_ix = indices.get_index(gather_pos as u32);

            for j in 0..axis {
                input_ix[j as usize] = out_ix[j as usize];
            }
            input_ix[axis as usize] = ax_ix as usize;
            for j in (axis as usize + 1)..r {
                input_ix[j] = out_ix[j + q as usize - 1];
            }

            values[i] = self.get(&input_ix);

            increment_index(&mut out_ix, &result_shape);
        }

        Tensor {
            shape: result_shape,
            strides: result_strides,
            size: result_size,
            values
        }
    }

    pub fn floor(&self) -> Tensor {
        self.unary_op(|x: f32| x.floor())
    }

    pub fn ceil(&self) -> Tensor {
        self.unary_op(|x: f32| x.ceil())
    }
}

impl Add for Tensor {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        return self.binary_op(&other, |x: f32, y: f32| x + y);
    }
}

impl Sub for Tensor {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        return self.binary_op(&other, |x: f32, y: f32| x - y);
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if !compare_shapes(&self.shape, &other.shape) {
            return false;
        }

        for i in 0..self.size {
            if self.values[i] != other.get_values()[i] {
                return false;
            }
        }

        return true;
    }
}

impl PartialOrd for Tensor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !compare_shapes(&self.shape, &other.shape) {
            return None;
        }

        if self.size == 0 {
            return Some(Ordering::Equal);
        }

        let mut val: Option<Ordering>;

        if self.values[0] > other.get_values()[0] {
            val = Some(Ordering::Greater);
        } else if self.values[0] < other.get_values()[0] {
            val = Some(Ordering::Less);
        } else {
            val = Some(Ordering::Equal);
        }

        for i in 1..self.size {
            let diff = self.values[i] - other.get_values()[i];
            if diff < 0. {
                if val != Some(Ordering::Less) {
                    val = None;
                    break;
                }
            } else if diff > 0. {
                if val != Some(Ordering::Greater) {
                    val = None;
                    break;
                }
            } else if val != Some(Ordering::Equal) {
                val = None;
                break;
            }
        }
        return val;
    }
}
