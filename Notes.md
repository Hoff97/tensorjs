- Xvfb -ac :99 -screen 0 1280x1024x16 & export DISPLAY=:99


- TODO:
  - Concat of sparse tensors along dense axis
  - Sparse-sparse matrix multiplication
  - Add toDense method to sparse tensor
  - Binary ops for:
    - WASM
    - GPU
