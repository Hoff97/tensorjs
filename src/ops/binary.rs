use crate::shape::*;
use crate::tensor::*;
use num_traits::int::PrimInt;
use num_traits::zero;
use num_traits::Float;
use num_traits::FromPrimitive;
use num_traits::Num;
use num_traits::ToPrimitive;

use std::cmp;

impl<DType> Tensor<DType>
where
    DType: Copy,
    DType: Num,
{
    #[inline]
    pub fn binary_op<F>(&self, other: &Tensor<DType>, op: F) -> Tensor<DType>
    where
        F: Fn(DType, DType) -> DType,
    {
        let mut result_shape = vec![0; self.rank()];
        for i in 0..self.rank() {
            result_shape[i] = cmp::max(self.get_dim_size(i), other.get_dim_size(i));
        }
        let result_size = get_size(&result_shape);
        let result_strides = compute_strides(&result_shape);

        let mut values: Vec<DType> = vec![zero(); result_size];

        let mut ix = vec![0; self.rank()];

        for i in 0..result_size {
            values[i] = op(self.get(&ix), other.get(&ix));

            increment_index(&mut ix, &result_shape);
        }

        Tensor::new(result_shape, result_strides, result_size, values)
    }
}

impl<DType> Tensor<DType>
where
    DType: Copy,
    DType: Num,
    DType: PartialOrd,
{
    pub fn addition(&self, other: &Tensor<DType>, alpha: DType, beta: DType) -> Tensor<DType> {
        return self.binary_op(other, |x: DType, y: DType| x * alpha + y * beta);
    }

    pub fn subtraction(&self, other: &Tensor<DType>, alpha: DType, beta: DType) -> Tensor<DType> {
        return self.binary_op(other, |x: DType, y: DType| x * alpha - y * beta);
    }

    pub fn multiply(&self, other: &Tensor<DType>, alpha: DType) -> Tensor<DType> {
        self.binary_op(other, |x: DType, y: DType| x * alpha * y)
    }

    pub fn divide(&self, other: &Tensor<DType>, alpha: DType) -> Tensor<DType> {
        self.binary_op(other, |x: DType, y: DType| x / y * alpha)
    }

    pub fn clip_backward(&self, min: DType, max: DType, grad: &Tensor<DType>) -> Tensor<DType> {
        self.binary_op(grad, |v: DType, g: DType| {
            if v < min || v > max {
                return zero();
            }
            return g;
        })
    }

    pub fn clip_min_backward(&self, min: DType, grad: &Tensor<DType>) -> Tensor<DType> {
        self.binary_op(grad, |v: DType, g: DType| {
            if v < min {
                return zero();
            }
            return g;
        })
    }

    pub fn clip_max_backward(&self, max: DType, grad: &Tensor<DType>) -> Tensor<DType> {
        self.binary_op(grad, |v: DType, g: DType| {
            if v > max {
                return zero();
            }
            return g;
        })
    }
}

impl<DType> Tensor<DType>
where
    DType: Clone,
    DType: Num,
    DType: PartialOrd,
    DType: Float,
    DType: FromPrimitive,
{
    pub fn power_float(&self, other: &Tensor<DType>) -> Tensor<DType> {
        self.binary_op(other, |x: DType, y: DType| x.powf(y))
    }

    pub fn bce(&self, other: &Tensor<DType>) -> Tensor<DType> {
        match DType::from_f32(1.0) {
            Some(one) => self.binary_op(other, |x: DType, y: DType| {
                if Some(y) == DType::from_f32(1.0) {
                    return -x.ln();
                } else {
                    return -((one - x).ln());
                }
            }),
            None => panic!("Encountered dtype that cant represent 1.0"),
        }
    }

    pub fn bce_back(&self, other: &Tensor<DType>) -> Tensor<DType> {
        match DType::from_f32(1.0) {
            Some(one) => self.binary_op(other, |x: DType, y: DType| {
                if Some(y) == DType::from_f32(1.0) {
                    return -one / x;
                } else {
                    return one / (one - x);
                }
            }),
            None => panic!("Encountered dtype that cant represent 1.0"),
        }
    }
}

impl<DType> Tensor<DType>
where
    DType: Clone,
    DType: Num,
    DType: PartialOrd,
    DType: PrimInt,
    DType: ToPrimitive,
{
    pub fn power_int(&self, other: &Tensor<DType>) -> Tensor<DType> {
        self.binary_op(other, |x: DType, y: DType| {
            x.pow(*y.to_u32().get_or_insert(0))
        })
    }
}
