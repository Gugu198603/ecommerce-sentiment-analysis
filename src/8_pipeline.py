import schedule
import time
import os
import subprocess
from datetime import datetime
from utils import setup_logging

logger = setup_logging()

def run_script(script_path):
    """辅助函数：运行 Python 脚本并捕获输出"""
    logger.info(f">>> 正在执行: {script_path}")
    try:
        # 使用 subprocess.run 同步执行，避免 shell 注入风险，获取标准输出
        result = subprocess.run(
            ["python", script_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"[{script_path}] 执行成功！")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[{script_path}] 执行失败: \n{e.stderr}")
        return False

def data_pipeline_job():
    """定义完整的数据更新流水线任务"""
    logger.info(f"========== 开始执行自动化数据更新流水线 | {datetime.now()} ==========")
    
    # 步骤1：爬取增量数据 (如果我们要增量更新，可以在 crawler 里改造，这里全量演示)
    # 注意：为了自动化运行不卡在 input 等待上，我们利用环境变量或命令行参数传递
    logger.info("1. 启动增量数据爬取...")
    run_script("src/1_crawler.py")
    
    # 步骤2：数据清洗与预处理
    logger.info("2. 启动数据清洗与预处理...")
    run_script("src/2_preprocess.py")
    
    # 步骤3：更新业务分析大盘与图表
    logger.info("3. 重新生成业务分析与可视化报告...")
    run_script("src/5_analysis.py")
    
    # 步骤4 (可选)：增量训练 GNN 防刷单模型
    logger.info("4. 更新 GNN 防作弊模型拓扑结构与分类结果...")
    run_script("src/9_gnn_antispam.py")
    
    logger.info(f"========== 自动化流水线执行完毕！业务大盘已最新 | {datetime.now()} ==========")

def main():
    logger.info("已启动后台自动化流水线服务 (Cron Job)。")
    logger.info("当前配置：每天凌晨 02:00 自动抓取并更新数据大盘。")
    
    # 配置定时任务：每天凌晨 2 点执行
    schedule.every().day.at("02:00").do(data_pipeline_job)
    
    # 为了演示方便，也可以配置为每 30 分钟执行一次：
    # schedule.every(30).minutes.do(data_pipeline_job)
    
    # 如果想一启动就立刻跑一遍，取消下面这行注释：
    # data_pipeline_job()
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60) # 每分钟检查一次
    except KeyboardInterrupt:
        logger.info("手动终止自动化流水线。")

if __name__ == "__main__":
    main()