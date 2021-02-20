use crate::tensor::*;
use num_traits::zero;
use num_traits::Float;
use num_traits::FromPrimitive;
use num_traits::Num;
use num_traits::PrimInt;
use num_traits::Signed;

impl<DType> Tensor<DType>
where
    DType: Copy,
    DType: Num,
{
    #[inline]
    fn unary_op<F>(&self, op: F) -> Tensor<DType>
    where
        F: Fn(DType) -> DType,
    {
        let mut values: Vec<DType> = vec![zero(); self.size];
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

impl<DType> Tensor<DType>
where
    DType: Copy,
    DType: Num,
    DType: PartialOrd,
{
    pub fn add_multiply_scalar(&self, factor: DType, add: DType) -> Tensor<DType> {
        self.unary_op(|x: DType| x * factor + add)
    }

    pub fn clip(&self, min: DType, max: DType) -> Tensor<DType> {
        self.unary_op(|x: DType| {
            if x > max {
                max
            } else if x < min {
                min
            } else {
                x
            }
        })
    }

    pub fn clip_min(&self, min: DType) -> Tensor<DType> {
        self.unary_op(|x: DType| if x < min { min } else { x })
    }

    pub fn clip_max(&self, max: DType) -> Tensor<DType> {
        self.unary_op(|x: DType| if x > max { max } else { x })
    }
}

impl<DType> Tensor<DType>
where
    DType: Copy,
    DType: Num,
    DType: Signed,
{
    pub fn abs(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.abs())
    }

    pub fn sign(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.signum())
    }

    pub fn negate(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.neg())
    }
}

impl<DType> Tensor<DType>
where
    DType: Copy,
    DType: Num,
    DType: Float,
    DType: FromPrimitive,
{
    pub fn exp(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.exp())
    }

    pub fn log(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.ln())
    }

    pub fn sqrt(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.sqrt())
    }

    pub fn sin(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.sin())
    }

    pub fn cos(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.cos())
    }

    pub fn tan(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.tan())
    }

    pub fn asin(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.asin())
    }

    pub fn acos(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.acos())
    }

    pub fn atan(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.atan())
    }

    pub fn sinh(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.sinh())
    }

    pub fn cosh(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.cosh())
    }

    pub fn tanh(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.tanh())
    }

    pub fn asinh(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.asinh())
    }

    pub fn acosh(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.acosh())
    }

    pub fn atanh(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.atanh())
    }

    pub fn sigmoid(&self) -> Tensor<DType> {
        match DType::from_f32(1.0) {
            Some(one) => self.unary_op(|x: DType| one / (one + (-x).exp())),
            None => panic!("Encountered DType that can not represent 1 in sigmoid"),
        }
    }

    pub fn floor(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.floor())
    }

    pub fn ceil(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.ceil())
    }

    pub fn round(&self) -> Tensor<DType> {
        self.unary_op(|x: DType| x.round())
    }

    pub fn power_scalar_float(&self, power: DType, factor: DType) -> Tensor<DType> {
        self.unary_op(|x: DType| x.powf(power) * factor)
    }

    pub fn hard_sigmoid(&self, alpha: DType, beta: DType) -> Tensor<DType> {
        match DType::from_f32(1.0) {
            Some(one) => self.unary_op(|x: DType| (alpha * x + beta).min(one).max(zero())),
            None => panic!("Encountered DType that can not represent 1 in hard_sigmoid"),
        }
    }
}

impl<DType> Tensor<DType>
where
    DType: Clone,
    DType: Num,
    DType: PartialOrd,
    DType: PrimInt,
{
    pub fn power_scalar_int(&self, power: u32, factor: DType) -> Tensor<DType> {
        self.unary_op(|x: DType| x.pow(power) * factor)
    }
}
