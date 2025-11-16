#!/bin/bash
# 遍历当前目录及子目录所有 .upf 文件
find . -type f -name "*.upf" | while read -r file; do
    # 提取目录和文件名
    dir=$(dirname "$file")
    base=$(basename "$file")
    # 转小写
    newbase=$(echo "$base" | tr '[:upper:]' '[:lower:]')
    # 如果新名字不同，就改名
    if [[ "$base" != "$newbase" ]]; then
        mv -v "$dir/$base" "$dir/$newbase"
    fi
done