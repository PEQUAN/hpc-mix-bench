#!/bin/bash

# 遍历当前目录下的每个子目录
for parent in */; do
  echo "➡️ 处理目录: $parent"

  mkdir -p "${parent}/all_cpp"

  # 在每个子目录中找 digit*_* 目录
  for dir in "${parent}"/digit*_*; do
    [ -d "$dir" ] || continue

    name=$(basename "$dir")        # digit1_3
    ij=${name#digit}              # 1_3
    i=${ij%%_*}                   # 1
    j=${ij##*_}                   # 3

    cpp_file=$(find "$dir" -maxdepth 1 -name "*.cpp" | head -n 1)

    if [ -f "$cpp_file" ]; then
      new_name="digit_${i}_${j}.cpp"
      cp "$cpp_file" "${parent}/all_cpp/$new_name"
    else
      echo "⚠️ $dir 没有 cpp 文件"
    fi
  done

  echo "✅ $parent 完成"
done

echo "🎉 全部处理完成"