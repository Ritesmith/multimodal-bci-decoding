#!/usr/bin/env python
"""
最简单的pip下载器 - 使用Python内置功能
"""
import sys
import os

def main():
    print("🚀 pip下载器")
    print("=" * 40)
    
    # 方法1: 尝试使用内置的urllib
    try:
        print("\n📥 尝试下载get-pip.py...")
        import urllib.request
        
        url = "https://bootstrap.pypa.io/get-pip.py"
        filename = "get-pip.py"
        
        print(f"URL: {url}")
        print(f"保存到: {os.path.abspath(filename)}")
        
        urllib.request.urlretrieve(url, filename)
        print("✅ 下载成功!")
        print(f"\n🔧 现在运行: python {filename}")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def fallback_instructions():
    """提供备用方案"""
    print("\n📋 手动下载步骤:")
    print("1. 打开浏览器")
    print("2. 访问: https://bootstrap.pypa.io/get-pip.py")
    print("3. 右键页面 -> 另存为")
    print("4. 保存到当前目录，文件名: get-pip.py")
    print("5. 运行: python get-pip.py")

if __name__ == "__main__":
    if main():
        print("\n🎯 下一步:")
        print("1. python get-pip.py")
        print("2. python -m pip install numpy matplotlib scipy")
        print("3. python test_fix.py")
        print("4. python main.py --mode experiment")
    else:
        fallback_instructions()