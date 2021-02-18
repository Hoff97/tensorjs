use crate::shape::*;
use crate::tensor::*;
use wasm_bindgen::prelude::*;

use std::cmp;

impl Tensor {
    #[inline]
    pub fn binary_op<F>(&self, other: &Tensor, op: F) -> Tensor
    where
        F: Fn(f32, f32) -> f32,
    {
        let mut result_shape = vec![0; self.rank()];
        for i in 0..self.rank() {
            result_shape[i] = cmp::max(self.get_dim_size(i), other.get_dim_size(i));
        }
        let result_size = get_size(&result_shape);
        let result_strides = compute_strides(&result_shape);

        let mut values: Vec<f32> = vec![0.0; result_size];

        let mut ix = vec![0; self.rank()];

        for i in 0..result_size {
            values[i] = op(self.get(&ix), other.get(&ix));

            increment_index(&mut ix, &result_shape);
        }

        Tensor::new(result_shape, result_strides, result_size, values)
    }
}

#[wasm_bindgen]
impl Tensor {
    pub fn addition(&self, other: &Tensor, alpha: f32, beta: f32) -> Tensor {
        return self.binary_op(other, |x: f32, y: f32| x * alpha + y * beta);
    }

    pub fn subtraction(&self, other: &Tensor, alpha: f32, beta: f32) -> Tensor {
        return self.binary_op(other, |x: f32, y: f32| x * alpha - y * beta);
    }

    pub fn multiply(&self, other: &Tensor, alpha: f32) -> Tensor {
        self.binary_op(other, |x: f32, y: f32| x * alpha * y)
    }

    pub fn divide(&self, other: &Tensor, alpha: f32) -> Tensor {
        self.binary_op(other, |x: f32, y: f32| x / y * alpha)
    }

    pub fn power(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, |x: f32, y: f32| x.powf(y))
    }

    pub fn bce(&self, other: &Tensor) -> Tensor {
        return self.binary_op(other, |x: f32, y: f32| {
            if y == 1.0 {
                return -x.ln();
            } else {
                return -((1.0 - x).ln());
            }
        });
    }

    pub fn bce_back(&self, other: &Tensor) -> Tensor {
        return self.binary_op(other, |x: f32, y: f32| {
            if y == 1.0 {
                return -1.0 / x;
            } else {
                return 1.0 / (1.0 - x);
            }
        });
    }

    pub fn clip_backward(&self, min: f32, max: f32, grad: &Tensor) -> Tensor {
        self.binary_op(grad, |v: f32, g: f32| {
            if v < min || v > max {
                return 0.0;
            }
            return g;
        })
    }

    pub fn clip_min_backward(&self, min: f32, grad: &Tensor) -> Tensor {
        self.binary_op(grad, |v: f32, g: f32| {
            if v < min {
                return 0.0;
            }
            return g;
        })
    }

    pub fn clip_max_backward(&self, max: f32, grad: &Tensor) -> Tensor {
        self.binary_op(grad, |v: f32, g: f32| {
            if v > max {
                return 0.0;
            }
            return g;
        })
    }
}
