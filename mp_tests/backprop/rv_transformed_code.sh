#!/bin/bash

# 遍历 i 和 j
for i in {1..4}; do
  for j in {1..10}; do
    dir="prec${i}_${j}"
    
    # 判断目录是否存在
    if [ -d "$dir" ]; then
      for file in "$dir"/*.c; do
        # 如果没有匹配到文件，会返回字面值，需判断
        [ -e "$file" ] || continue
        
        filename=$(basename "$file")
        newname="${dir}_${filename}"
        
        mv "$file" "$dir/$newname"
      done
    fi
  done
done