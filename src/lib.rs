pub mod utils;
pub mod shape;
pub mod tensor;

pub mod tests {
    pub mod shape;
    pub mod tensor {
        pub mod basic;
        pub mod tensor;
    }
}

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
