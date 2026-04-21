#!/bin/bash

mkdir -p all_cpp

for dir in digit*_*; do
  # 提取 i 和 j
  name=$(basename "$dir")          # digit1_3
  ij=${name#digit}                # 1_3
  i=${ij%%_*}                     # 1
  j=${ij##*_}                     # 3

  cpp_file=$(find "$dir" -maxdepth 1 -name "*.cpp" | head -n 1)

  if [ -f "$cpp_file" ]; then
    new_name="digit_${i}_${j}.cpp"
    cp "$cpp_file" "all_cpp/$new_name"
  else
    echo "⚠️ $dir 没有 cpp 文件"
  fi
done

echo "✅ 完成（自动扫描版）"