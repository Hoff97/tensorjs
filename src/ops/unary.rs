use crate::shape::*;
use crate::tensor::*;
use wasm_bindgen::prelude::*;

use std::cmp;

impl Tensor {
    #[inline]
    fn unary_op<F>(&self, op: F) -> Tensor
    where
        F: Fn(f32) -> f32,
    {
        let mut values: Vec<f32> = vec![0.0; self.size];
        for i in 0..self.size {
            values[i] = op(self.get_ix(i));
        }

        Tensor::new(
            self.get_sh().to_vec(),
            self.get_strides().to_vec(),
            self.size,
            values,
        )
    }
}

#[wasm_bindgen]
impl Tensor {
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

    pub fn sin(&self) -> Tensor {
        self.unary_op(|x: f32| x.sin())
    }

    pub fn cos(&self) -> Tensor {
        self.unary_op(|x: f32| x.cos())
    }

    pub fn tan(&self) -> Tensor {
        self.unary_op(|x: f32| x.tan())
    }

    pub fn asin(&self) -> Tensor {
        self.unary_op(|x: f32| x.asin())
    }

    pub fn acos(&self) -> Tensor {
        self.unary_op(|x: f32| x.acos())
    }

    pub fn atan(&self) -> Tensor {
        self.unary_op(|x: f32| x.atan())
    }

    pub fn sinh(&self) -> Tensor {
        self.unary_op(|x: f32| x.sinh())
    }

    pub fn cosh(&self) -> Tensor {
        self.unary_op(|x: f32| x.cosh())
    }

    pub fn tanh(&self) -> Tensor {
        self.unary_op(|x: f32| x.tanh())
    }

    pub fn asinh(&self) -> Tensor {
        self.unary_op(|x: f32| x.asinh())
    }

    pub fn acosh(&self) -> Tensor {
        self.unary_op(|x: f32| x.acosh())
    }

    pub fn atanh(&self) -> Tensor {
        self.unary_op(|x: f32| x.atanh())
    }

    pub fn sigmoid(&self) -> Tensor {
        self.unary_op(|x: f32| 1.0 / (1.0 + (-x).exp()))
    }

    pub fn hard_sigmoid(&self, alpha: f32, beta: f32) -> Tensor {
        self.unary_op(|x: f32| (alpha * x + beta).min(1.0).max(0.0))
    }

    pub fn sign(&self) -> Tensor {
        self.unary_op(|x: f32| {
            if x < 0. {
                -1.
            } else if x == 0.0 {
                0.
            } else {
                1.
            }
        })
    }

    pub fn negate(&self) -> Tensor {
        self.unary_op(|x: f32| -x)
    }

    pub fn power_scalar(&self, power: f32, factor: f32) -> Tensor {
        self.unary_op(|x: f32| x.powf(power) * factor)
    }

    pub fn add_multiply_scalar(&self, factor: f32, add: f32) -> Tensor {
        self.unary_op(|x: f32| x * factor + add)
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

    pub fn floor(&self) -> Tensor {
        self.unary_op(|x: f32| x.floor())
    }

    pub fn ceil(&self) -> Tensor {
        self.unary_op(|x: f32| x.ceil())
    }

    pub fn round(&self) -> Tensor {
        self.unary_op(|x: f32| x.round())
    }
}
