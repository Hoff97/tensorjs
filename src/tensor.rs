use crate::shape::compare_shapes;
use crate::shape::compute_strides;
use crate::shape::get_size;
use crate::shape::index_to_pos;
use num_traits::abs;
use num_traits::FromPrimitive;
use num_traits::Num;
use num_traits::Signed;
use std::cmp::Ordering;
use std::ops::Add;
use std::ops::Sub;

#[derive(Debug, Clone, std::marker::Copy)]
pub struct Tensor<DType> {
    shape: Vec<usize>,
    strides: Vec<usize>,
    pub size: usize,
    values: Vec<DType>,
}

impl<DType> Tensor<DType>
where
    DType: Clone,
{
    pub fn new(
        shape: Vec<usize>,
        strides: Vec<usize>,
        size: usize,
        values: Vec<DType>,
    ) -> Tensor<DType> {
        Tensor {
            shape,
            strides,
            size,
            values,
        }
    }

    pub fn get_sh(&self) -> &Vec<usize> {
        return &self.shape;
    }

    pub fn get_dim_size(&self, dim: usize) -> usize {
        return self.shape[dim];
    }

    pub fn rank(&self) -> usize {
        return self.shape.len();
    }

    pub fn get_strides(&self) -> &Vec<usize> {
        return &self.strides;
    }

    pub fn get_strides_at(&self, ix: usize) -> usize {
        return self.strides[ix];
    }

    pub fn get_values(&self) -> &Vec<DType> {
        return &self.values;
    }

    pub fn get(&self, index: &Vec<usize>) -> DType {
        let pos = index_to_pos(index, self.get_strides());
        return self.values[pos];
    }

    pub fn get_ix(&self, index: usize) -> DType {
        return self.values[index];
    }

    pub fn set(&mut self, index: &Vec<usize>, value: DType) {
        let pos = index_to_pos(index, &self.strides);
        self.values[pos] = value;
    }

    pub fn new_from_shape(shape: &Vec<usize>, values: &Vec<DType>) -> Tensor<DType> {
        let strides = compute_strides(shape);
        let size = get_size(shape);

        Tensor::new(shape.to_vec(), strides, size, values.to_vec())
    }

    pub fn constant(shape: &Vec<usize>, value: DType) -> Tensor<DType> {
        let strides = compute_strides(shape);
        let size = get_size(shape);

        let values = vec![value; size];

        Tensor::new(shape.to_vec(), strides, size, values)
    }
}

impl<DType> Tensor<DType>
where
    DType: Clone,
    DType: Num,
    DType: Signed,
    DType: PartialOrd,
{
    pub fn compare(&self, other: &Self, delta: DType) -> bool {
        if !compare_shapes(self.get_sh(), other.get_sh()) {
            return false;
        }

        for i in 0..self.size {
            if abs(self.get_ix(i) - other.get_ix(i)) > delta {
                return false;
            }
        }

        return true;
    }
}

impl<DType> Add for Tensor<DType> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        return self.binary_op(&other, |x: f32, y: f32| x + y);
    }
}

impl<DType> Sub for Tensor<DType> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        return self.binary_op(&other, |x: f32, y: f32| x - y);
    }
}

impl<DType> PartialEq for Tensor<DType>
where
    DType: Clone,
    DType: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if !compare_shapes(self.get_sh(), other.get_sh()) {
            return false;
        }

        for i in 0..self.size {
            if self.get_ix(i) != other.get_values()[i] {
                return false;
            }
        }

        return true;
    }
}

impl<DType> PartialOrd for Tensor<DType>
where
    DType: Clone,
    DType: PartialOrd,
    DType: Num,
    DType: FromPrimitive,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !compare_shapes(self.get_sh(), other.get_sh()) {
            return None;
        }

        if self.size == 0 {
            return Some(Ordering::Equal);
        }

        let mut val: Option<Ordering>;

        if self.get_ix(0) > other.get_ix(0) {
            val = Some(Ordering::Greater);
        } else if self.get_ix(0) < other.get_ix(0) {
            val = Some(Ordering::Less);
        } else {
            val = Some(Ordering::Equal);
        }

        for i in 1..self.size {
            let diff = self.get_ix(i) - other.get_ix(i);
            if Some(diff) < DType::from_i32(0) {
                if val != Some(Ordering::Less) {
                    val = None;
                    break;
                }
            } else if Some(diff) > DType::from_i32(0) {
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
