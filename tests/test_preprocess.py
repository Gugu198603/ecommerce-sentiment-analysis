import sys
import os
import pytest
import importlib.util

# 将 src 目录加入路径，以便导入模块
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

# 因为文件名以数字开头，不能直接用 import
preprocess_module = importlib.import_module("2_preprocess")
clean_text = preprocess_module.clean_text

def test_clean_text_normal():
    assert clean_text("这件商品真的很好用，强烈推荐！") == "这件商品真的很好用，强烈推荐！"

def test_clean_text_with_url():
    assert clean_text("快看这个链接 http://jd.com/123 很便宜") == "快看这个链接 很便宜"

def test_clean_text_with_html():
    assert clean_text("<div>质量不错<br>就是物流太慢</div>") == "质量不错就是物流太慢"

def test_clean_text_with_special_chars():
    assert clean_text("很好用~！！！@#%&*") == "很好用！！！"

def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""