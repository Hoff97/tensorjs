use crate::dtype::u32::TensorU32;
use crate::shape::compute_strides;
use crate::shape::get_size;
use crate::tensor::Tensor;
use js_sys::Float32Array;
use js_sys::Int32Array;
use js_sys::Uint32Array;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct TensorI32 {
    tensor: Tensor<i32>,
}

type Elem = i32;
type Sel = TensorI32;

#[wasm_bindgen]
impl TensorI32 {
    pub fn create(shape: Uint32Array, values: Int32Array) -> TensorI32 {
        let mut _shape: Vec<usize> = vec![0; shape.length() as usize];
        for i in 0.._shape.len() {
            _shape[i] = shape.get_index(i as u32) as usize;
        }

        let strides = compute_strides(&_shape);
        let size = get_size(&_shape);

        let mut _values: Vec<i32> = vec![0; values.length() as usize];
        for i in 0.._values.len() {
            _values[i] = values.get_index(i as u32);
        }

        TensorI32 {
            tensor: Tensor::new(_shape, strides, size, _values),
        }
    }

    pub fn create_constant(shape: Uint32Array, value: i32) -> TensorI32 {
        let mut _shape: Vec<usize> = vec![0; shape.length() as usize];
        for i in 0.._shape.len() {
            _shape[i] = shape.get_index(i as u32) as usize;
        }

        let strides = compute_strides(&_shape);
        let size = get_size(&_shape);

        let values = vec![value; size];

        TensorI32 {
            tensor: Tensor::new(_shape, strides, size, values),
        }
    }

    pub fn get_vals(&self) -> Int32Array {
        let arr = Int32Array::new_with_length(self.tensor.size as u32);

        for i in 0..self.tensor.size {
            arr.set_index(i as u32, self.tensor.get_ix(i));
        }

        return arr;
    }

    pub fn get_shape(&self) -> Uint32Array {
        let arr = Uint32Array::new_with_length(self.tensor.rank() as u32);

        for i in 0..self.tensor.rank() {
            arr.set_index(i as u32, self.tensor.get_dim_size(i) as u32);
        }

        return arr;
    }

    pub fn power_scalar(&self, power: u32, factor: i32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.power_scalar_int(power, factor),
        }
    }

    pub fn abs(&self) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.abs(),
        }
    }

    pub fn sign(&self) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.sign(),
        }
    }

    pub fn negate(&self) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.negate(),
        }
    }

    pub fn add_multiply_scalar(&self, factor: i32, add: i32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.add_multiply_scalar(factor, add),
        }
    }

    pub fn clip(&self, min: i32, max: i32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.clip(min, max),
        }
    }

    pub fn clip_min(&self, min: i32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.clip_min(min),
        }
    }

    pub fn clip_max(&self, max: i32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.clip_max(max),
        }
    }

    pub fn power(&self, other: &TensorI32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.power_int(&other.tensor),
        }
    }

    pub fn addition(&self, other: &TensorI32, alpha: i32, beta: i32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.addition(&other.tensor, alpha, beta),
        }
    }

    pub fn subtraction(&self, other: &TensorI32, alpha: i32, beta: i32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.subtraction(&other.tensor, alpha, beta),
        }
    }

    pub fn multiply(&self, other: &TensorI32, alpha: i32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.multiply(&other.tensor, alpha),
        }
    }

    pub fn divide(&self, other: &TensorI32, alpha: i32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.divide(&other.tensor, alpha),
        }
    }

    pub fn clip_backward(&self, min: i32, max: i32, grad: &TensorI32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.clip_backward(min, max, &grad.tensor),
        }
    }

    pub fn clip_min_backward(&self, min: i32, grad: &TensorI32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.clip_min_backward(min, &grad.tensor),
        }
    }

    pub fn clip_max_backward(&self, max: i32, grad: &TensorI32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.clip_max_backward(max, &grad.tensor),
        }
    }

    pub fn sum(&self, axes: Uint32Array, keep_dims: bool) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.sum(axes, keep_dims),
        }
    }

    pub fn sum_square(&self, axes: Uint32Array, keep_dims: bool) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.sum_square(axes, keep_dims),
        }
    }

    pub fn product(&self, axes: Uint32Array, keep_dims: bool) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.product(axes, keep_dims),
        }
    }

    pub fn max(&self, axes: Uint32Array, keep_dims: bool) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.max(axes, keep_dims),
        }
    }

    pub fn min(&self, axes: Uint32Array, keep_dims: bool) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.min(axes, keep_dims),
        }
    }

    pub fn reduce_mean(&self, axes: Uint32Array, keep_dims: bool) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.reduce_mean(axes, keep_dims),
        }
    }

    pub fn reduce_mean_square(&self, axes: Uint32Array, keep_dims: bool) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.reduce_mean_square(axes, keep_dims),
        }
    }

    pub fn conv(
        &self,
        kernel: &TensorI32,
        dilations: Uint32Array,
        group: u32,
        pads: Uint32Array,
        strides: Uint32Array,
        activation: u32,
    ) -> TensorI32 {
        TensorI32 {
            tensor: self
                .tensor
                .conv(&kernel.tensor, dilations, group, pads, strides, activation),
        }
    }

    pub fn conv_with_bias(
        &self,
        kernel: &TensorI32,
        bias: &TensorI32,
        dilations: Uint32Array,
        group: u32,
        pads: Uint32Array,
        strides: Uint32Array,
        activation: u32,
    ) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.conv_with_bias(
                &kernel.tensor,
                &bias.tensor,
                dilations,
                group,
                pads,
                strides,
                activation,
            ),
        }
    }

    pub fn conv_transpose(
        &self,
        kernel: &TensorI32,
        dilations: Uint32Array,
        group: u32,
        pads: Uint32Array,
        strides: Uint32Array,
    ) -> TensorI32 {
        TensorI32 {
            tensor: self
                .tensor
                .conv_transpose(&kernel.tensor, dilations, group, pads, strides),
        }
    }

    pub fn average_pool(
        &self,
        kernel_shape: Uint32Array,
        pads: Uint32Array,
        strides: Uint32Array,
        include_pad: bool,
    ) -> TensorI32 {
        TensorI32 {
            tensor: self
                .tensor
                .average_pool(kernel_shape, pads, strides, include_pad),
        }
    }

    // Mode: 0 == constant, 1 == reflect, 2 == edge
    pub fn pad(&self, pads: Uint32Array, mode: i32, value: i32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.pad(pads, mode, value),
        }
    }

    pub fn upsample(&self, scales: Float32Array) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.upsample(scales),
        }
    }

    pub fn matmul(&self, other: &TensorI32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.matmul(&other.tensor),
        }
    }

    pub fn gemm(
        &self,
        other: &TensorI32,
        a_transpose: bool,
        b_transpose: bool,
        alpha: i32,
    ) -> TensorI32 {
        TensorI32 {
            tensor: self
                .tensor
                .gemm(&other.tensor, a_transpose, b_transpose, alpha),
        }
    }

    pub fn gemm_with_c(
        &self,
        other: &TensorI32,
        a_transpose: bool,
        b_transpose: bool,
        alpha: i32,
        c: &TensorI32,
        beta: i32,
    ) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.gemm_with_c(
                &other.tensor,
                a_transpose,
                b_transpose,
                alpha,
                &c.tensor,
                beta,
            ),
        }
    }

    pub fn set_values(&self, values: &TensorI32, starts: Uint32Array) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.set_values(&values.tensor, starts),
        }
    }

    pub fn reshape(&self, shape: Uint32Array) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.reshape(shape),
        }
    }

    pub fn concat(&self, other: &TensorI32, axes: u32) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.concat(&other.tensor, axes),
        }
    }

    pub fn transpose(&self, permutation: Uint32Array) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.transpose(permutation),
        }
    }

    pub fn repeat(&self, repeats: Uint32Array) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.repeat(repeats),
        }
    }

    pub fn expand(&self, shape: Uint32Array) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.expand(shape),
        }
    }

    pub fn copy(&self) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.copy(),
        }
    }

    // Mode: 0 == constant, 1 == reflect, 2 == edge
    pub fn gather(&self, axis: i32, indices: Uint32Array, indice_shape: Uint32Array) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.gather(axis, indices, indice_shape),
        }
    }

    pub fn slice(
        &self,
        starts: Uint32Array,
        ends: Uint32Array,
        axis: Uint32Array,
        steps: Int32Array,
    ) -> TensorI32 {
        TensorI32 {
            tensor: self.tensor.slice(starts, ends, axis, steps),
        }
    }

    pub fn matmul_sparse_dense(&self, indices: &TensorU32, b: &TensorI32, m: usize) -> TensorI32 {
        TensorI32 {
            tensor: self
                .tensor
                .matmul_sparse_dense(indices.get_tensor(), &b.tensor, m),
        }
    }

    pub fn add_sparse_dense(
        &self,
        indices: &TensorU32,
        b: &Sel,
        result_shape: Uint32Array,
        alpha: Elem,
        beta: Elem,
    ) -> Self {
        Self {
            tensor: self.tensor.add_sparse_dense(
                indices.get_tensor(),
                &b.tensor,
                result_shape,
                alpha,
                beta,
            ),
        }
    }

    pub fn subtract_sparse_dense(
        &self,
        indices: &TensorU32,
        b: &Sel,
        result_shape: Uint32Array,
        alpha: Elem,
        beta: Elem,
    ) -> Self {
        Self {
            tensor: self.tensor.subtract_sparse_dense(
                indices.get_tensor(),
                &b.tensor,
                result_shape,
                alpha,
                beta,
            ),
        }
    }

    pub fn multiply_sparse_dense(
        &self,
        indices: &TensorU32,
        b: &Sel,
        result_shape: Uint32Array,
        alpha: Elem,
    ) -> Self {
        Self {
            tensor: self.tensor.multiply_sparse_dense(
                indices.get_tensor(),
                &b.tensor,
                result_shape,
                alpha,
            ),
        }
    }

    pub fn divide_sparse_dense(
        &self,
        indices: &TensorU32,
        b: &Sel,
        result_shape: Uint32Array,
        alpha: Elem,
    ) -> Self {
        Self {
            tensor: self.tensor.divide_sparse_dense(
                indices.get_tensor(),
                &b.tensor,
                result_shape,
                alpha,
            ),
        }
    }

    pub fn add_sparse_sparse(
        &self,
        indices: &TensorU32,
        b_indices: &TensorU32,
        b_values: &Sel,
        result_shape: Uint32Array,
        alpha: Elem,
        beta: Elem,
    ) -> Self {
        Self {
            tensor: self.tensor.add_sparse_sparse(
                indices.get_tensor(),
                b_indices.get_tensor(),
                &b_values.tensor,
                result_shape,
                alpha,
                beta,
            ),
        }
    }
}
