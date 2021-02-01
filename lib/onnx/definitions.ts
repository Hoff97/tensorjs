// Attribute types
export const ATTRIBUTE_UNDEFINED = 0;
export const ATTRIBUTE_FLOAT = 1; // Float32
export const ATTRIBUTE_INT = 2; // Int64
export const ATTRIBUTE_STRING = 3;
export const ATTRIBUTE_TENSOR = 4;
export const ATTRIBUTE_GRAPH = 5;
export const ATTRIBUTE_SPARSE_TENSOR = 11;

export const ATTRIBUTE_FLOATS = 6;
export const ATTRIBUTE_INTS = 7;
export const ATTRIBUTE_STRINGS = 8;
export const ATTRIBUTE_TENSORS = 9;
export const ATTRIBUTE_GRAPHS = 10;
export const ATTRIBUTE_SPARSE_TENSORS = 12;

// Tensor types
export const TENSOR_FLOAT = 1; // float  (32 bits)
export const TENSOR_UINT8 = 2; // uint8_t
export const TENSOR_INT8 = 3; // int8_t
export const TENSOR_UINT16 = 4; // uint16_t
export const TENSOR_INT16 = 5; // int16_t
export const TENSOR_INT32 = 6; // int32_t
export const TENSOR_INT64 = 7; // int64_t
export const TENSOR_STRING = 8; // string
export const TENSOR_BOOL = 9; // bool

export const TENSOR_FLOAT16 = 10;
export const TENSOR_DOUBLE = 11;
export const TENSOR_UINT32 = 12;
export const TENSOR_UINT64 = 13;
export const TENSOR_COMPLEX64 = 14; // complex with float32 real and imaginary components
export const TENSOR_COMPLEX128 = 15; // complex with float64 real and imaginary components

// Non-IEEE floating-point format based on IEEE754 single-precision
// floating-point number truncated to 16 bits.
// This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
export const TENSOR_BFLOAT16 = 16;
