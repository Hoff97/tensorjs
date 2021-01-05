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
    pub fn pool_continuous<F>(&self, axes: &Vec<usize>, op: F) -> Tensor where F: Fn(f32,f32) -> f32 {
        let result_rank = self.shape.len() - axes.len() as usize;
        
        let mut result_shape = vec![0; result_rank];
        let mut result_size = 1;
        let mut sum_size = 1;
        let mut axes_ix = 0;
        let mut result_ix = 0;
        for i in 0..self.shape.len() {
            if axes_ix < axes.len() && axes[axes_ix] as usize == i {
                axes_ix += 1;
                sum_size *= self.shape[i];
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
                for k in 1..sum_size {
                    res = op(self.values[input_start_ix + j + k*step_size], res);
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
    pub fn _pool<F>(&self, axes: &Vec<usize>, op: F) -> Tensor where F: Fn(f32,f32) -> f32 {
        let result_rank = self.shape.len() - axes.len() as usize;
        if result_rank == 0 {
            let mut value = self.values[0];
            for i in 1..self.size {
                value = op(value, self.values[i]);
            }

            return Tensor {
                values: vec![value],
                shape: vec![1],
                size: 1,
                strides: vec![1]
            }
        }

        if self.axes_continuous(axes) {
            return self.pool_continuous(axes, op);
        }

        let mut result_shape = vec![0; result_rank];
        let mut result_size = 1;
        let mut axes_ix = 0;
        let mut result_ix = 0;
        let mut res_ix_map = vec![0; result_rank];
        for i in 0..self.shape.len() {
            if axes_ix < axes.len() && axes[axes_ix] as usize == i {
                axes_ix += 1;
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
                initialized[res_ix] = true;
            } else {
                values[res_ix] = op(self.values[i], values[res_ix]);
            }

            increment_index(&mut input_index, &self.shape);
        }

        Tensor {
            values,
            shape: result_shape,
            size: result_size,
            strides: result_strides
        }
    }

    pub fn _sum(&self, axes: &Vec<usize>) -> Tensor {
        return self._pool(axes, |x: f32, y: f32| x+y)
    }

    pub fn _product(&self, axes: &Vec<usize>) -> Tensor {
        return self._pool(axes, |x: f32, y: f32| x*y)
    }

    pub fn _max(&self, axes: &Vec<usize>) -> Tensor {
        return self._pool(axes, |x: f32, y: f32| x.max(y))
    }

    pub fn _min(&self, axes: &Vec<usize>) -> Tensor {
        return self._pool(axes, |x: f32, y: f32| x.min(y))
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
                  
        let R = conv_output_size(D, W, _pads, _dilations, _strides);
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

    pub fn sum(&self, axes: Uint32Array) -> Tensor {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._sum(&ax);
    }

    pub fn product(&self, axes: Uint32Array) -> Tensor {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._product(&ax);
    }

    pub fn max(&self, axes: Uint32Array) -> Tensor {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._max(&ax);
    }

    pub fn min(&self, axes: Uint32Array) -> Tensor {
        let mut ax: Vec<usize> = vec![0; axes.length() as usize];
        for i in 0..axes.length() {
            ax[i as usize] = axes.get_index(i) as usize;
        }
        return self._min(&ax);
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

    pub fn reshape(&self, shape: Uint32Array) -> Tensor {
        let mut sh: Vec<usize> = vec![0; shape.length() as usize];
        for i in 0..shape.length() {
            sh[i as usize] = shape.get_index(i) as usize;
        }
        return self._reshape(&sh);
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
