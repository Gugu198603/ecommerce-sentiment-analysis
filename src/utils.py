import os
import logging
from pathlib import Path

# 设置项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 目录配置
DIRS = {
    'raw_data': PROJECT_ROOT / 'data' / 'raw',
    'processed_data': PROJECT_ROOT / 'data' / 'processed',
    'models': PROJECT_ROOT / 'models' / 'sentiment_model',
    'figures': PROJECT_ROOT / 'results' / 'figures',
    'reports': PROJECT_ROOT / 'results' / 'reports'
}

def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def ensure_dirs():
    """确保所有必要的目录存在"""
    for dir_path in DIRS.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        # 创建 .gitkeep 保持空目录在 git 中可见
        (dir_path / '.gitkeep').touch(exist_ok=True)

if __name__ == '__main__':
    ensure_dirs()
    print("项目目录结构初始化完成！")