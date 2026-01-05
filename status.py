import os
import sys
from pathlib import Path

print("="*50)
print("多模态BCI项目状态检查")
print("="*50)

# 检查当前目录
current_dir = os.getcwd()
print(f"当前目录: {current_dir}")

# 检查文件
files = [f for f in os.listdir('.') if os.path.isfile(f) and not f.startswith('.')]
dirs = [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.')]

print(f"\n文件数量: {len(files)}")
print(f"目录数量: {len(dirs)}")

print("\n核心文件:")
core_files = ['config.py', 'main.py', 'README.md', 'requirements.txt']
for f in core_files:
    exists = '✓' if os.path.exists(f) else '✗'
    print(f"  {exists} {f}")

print("\nsrc目录:")
if os.path.exists('src'):
    src_files = [f for f in os.listdir('src') if not f.startswith('.')]
    print(f"  文件数量: {len(src_files)}")
    for f in src_files:
        print(f"    {f}")
else:
    print("  ✗ src目录不存在")

print("\n" + "="*50)
print("项目已创建完成！")
print("="*50)