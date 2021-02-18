use crate::shape::compare_shapes;
use crate::shape::compute_strides;
use crate::shape::get_size;
use crate::shape::index_to_pos;
use js_sys::Float32Array;
use js_sys::Uint32Array;
use std::cmp::Ordering;
use std::ops::Add;
use std::ops::Sub;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    pub size: usize,
    values: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, strides: Vec<usize>, size: usize, values: Vec<f32>) -> Tensor {
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

    pub fn get_values(&self) -> &Vec<f32> {
        return &self.values;
    }

    pub fn get(&self, index: &Vec<usize>) -> f32 {
        let pos = index_to_pos(index, self.get_strides());
        return self.values[pos];
    }

    pub fn get_ix(&self, index: usize) -> f32 {
        return self.values[index];
    }

    pub fn set(&mut self, index: &Vec<usize>, value: f32) {
        let pos = index_to_pos(index, &self.strides);
        self.values[pos] = value;
    }

    pub fn new_from_shape(shape: &Vec<usize>, values: &Vec<f32>) -> Tensor {
        let strides = compute_strides(shape);
        let size = get_size(shape);

        Tensor::new(shape.to_vec(), strides, size, values.to_vec())
    }

    pub fn constant(shape: &Vec<usize>, value: f32) -> Tensor {
        let strides = compute_strides(shape);
        let size = get_size(shape);

        let values = vec![value; size];

        Tensor::new(shape.to_vec(), strides, size, values)
    }

    pub fn compare(&self, other: &Self, delta: f32) -> bool {
        if !compare_shapes(self.get_sh(), other.get_sh()) {
            return false;
        }

        for i in 0..self.size {
            if (self.get_ix(i) - other.get_ix(i)).abs() > delta {
                return false;
            }
        }

        return true;
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

        Tensor::new(_shape, strides, size, _values)
    }

    pub fn create_constant(shape: Uint32Array, value: f32) -> Tensor {
        let mut _shape: Vec<usize> = vec![0; shape.length() as usize];
        for i in 0.._shape.len() {
            _shape[i] = shape.get_index(i as u32) as usize;
        }

        let strides = compute_strides(&_shape);
        let size = get_size(&_shape);

        let values = vec![value; size];

        Tensor::new(_shape, strides, size, values)
    }

    pub fn get_vals(&self) -> Float32Array {
        let arr = Float32Array::new_with_length(self.size as u32);

        for i in 0..self.size {
            arr.set_index(i as u32, self.get_ix(i));
        }

        return arr;
    }

    pub fn get_shape(&self) -> Uint32Array {
        let arr = Uint32Array::new_with_length(self.rank() as u32);

        for i in 0..self.rank() {
            arr.set_index(i as u32, self.get_dim_size(i) as u32);
        }

        return arr;
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

impl PartialOrd for Tensor {
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
