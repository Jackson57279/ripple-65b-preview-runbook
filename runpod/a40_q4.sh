#!/bin/bash
# One A40 (48 GB). Stock llama.cpp cannot load model_type ricursive.
# Q4_K_M is 36.13 GiB and fits. llama.cpp itself is not part of this repo.
set -euo pipefail
cd /workspace
curl -fsSL -o ricursive.patch \
  https://huggingface.co/mrfakename/Ripple-65B-Preview-GGUF/resolve/main/ricursive.patch
curl -fL -o Ripple-65B-Preview-Q4_K_M.gguf \
  https://huggingface.co/mrfakename/Ripple-65B-Preview-GGUF/resolve/main/Q4_K_M/Ripple-65B-Preview-Q4_K_M.gguf
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git fetch --depth 1 origin 6f1e6543704bc5b080c7a98e7b6f707b31f6ed51
git checkout 6f1e6543704bc5b080c7a98e7b6f707b31f6ed51
git apply /workspace/ricursive.patch
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j "$(nproc)" --target llama-cli
./build/bin/llama-cli \
  -m /workspace/Ripple-65B-Preview-Q4_K_M.gguf \
  --jinja -c 2048 -n 256 --temp 0 -ngl 99 \
  -p "Write a complete Python file named hello_ripple.py that prints the sum of 2 and 3. Output only the file."
