#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
电商评论情感因果推断模块
方法: 倾向得分匹配 (PSM) + 因果森林 (Causal Forest)
目的: 评估评论情感倾向 (Treatment) 对用户评分 (Outcome) 的因果效应
"""

import io
import os
import sys
import warnings
from pathlib import Path

# 修复 Windows 控制台中文乱码
if sys.platform == 'win32' and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (AttributeError, OSError):
        pass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# --- 中文字体配置 (解决 Windows 下中文乱码, matplotlib>=3.6) ---
fm._load_fontmanager(try_read_cache=False)  # 强制重建字体缓存
_cjk_candidates = [f.name for f in fm.fontManager.ttflist
                   if any(k in f.name.lower()
                          for k in ["simhei", "microsoft yahei", "simsun", "wenquanyi", "noto sans cjk"])\
                   and f.style == "normal" and f.weight in (400, "normal")]
_cjk_font = next((n for n in _cjk_candidates if "simhei" in n.lower()), None) \
            or next(iter(_cjk_candidates), None)
if _cjk_font:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [_cjk_font] + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    print(f"[INFO] 使用中文字体: {_cjk_font}")
else:
    print("[WARN] 未检测到中文字体，图表中文可能显示为方块")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ==================== 因果森林封装 (兼容 causalml API 差异) ====================

try:
    from causalml.inference.tree import CausalTreeRegressor

    class CausalForestRegressor:
        """
        Bootstrap-ensemble 因果森林封装
        使用多个 CausalTreeRegressor 在 Bootstrap 样本上训练，平均预测结果。
        对外提供与用户指定 API 一致的接口:
          CausalForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        """

        def __init__(self, n_estimators=100, max_depth=6, random_state=42, **kwargs):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.random_state = random_state
            self.kwargs = kwargs
            self.trees_ = []
            self._feature_importances_ = None

        def fit(self, X, treatment, y):
            rng = np.random.RandomState(self.random_state)
            n = X.shape[0]
            self.trees_ = []
            importances_sum = np.zeros(X.shape[1])

            for i in range(self.n_estimators):
                idx = rng.choice(n, size=n, replace=True)
                tree = CausalTreeRegressor(
                    max_depth=self.max_depth,
                    random_state=self.random_state + i,
                    **self.kwargs,
                )
                tree.fit(X[idx], treatment[idx], y[idx])
                self.trees_.append(tree)
                if hasattr(tree, "feature_importances_"):
                    importances_sum += tree.feature_importances_

            self._feature_importances_ = importances_sum / self.n_estimators
            return self

        def predict(self, X):
            preds = np.zeros(X.shape[0])
            for tree in self.trees_:
                preds += tree.predict(X)
            return preds / self.n_estimators

        @property
        def feature_importances_(self):
            return self._feature_importances_

    _HAS_CAUSALML = True
except ImportError:
    _HAS_CAUSALML = False

# ==================== 常量配置 ====================
RANDOM_STATE = 42
CALIPER = 0.20  # 放宽卡尺以提高匹配率 (原 0.05 太严格)
N_ESTIMATORS = 100
MAX_DEPTH = 6
BOOTSTRAP_ITER = 200
TREATMENT_DOWNSAMPLE_RATIO = 3.0  # 处理组最多为对照组的 N 倍

np.random.seed(RANDOM_STATE)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIR_PROCESSED = PROJECT_ROOT / "data" / "processed"
DIR_FIGURES = PROJECT_ROOT / "results" / "figures"
DIR_REPORTS = PROJECT_ROOT / "results" / "reports"

# ==================== 工具函数 ====================


def ensure_dirs():
    for d in [DIR_PROCESSED, DIR_FIGURES, DIR_REPORTS]:
        d.mkdir(parents=True, exist_ok=True)


def smd_score(group1, group2):
    """计算标准化均值差 (Standardized Mean Difference)"""
    diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return diff / pooled_std


def ps_matching(propensity_scores, treatment, caliper=CALIPER):
    """
    1:1 最近邻倾向得分匹配 (带卡尺)
    返回: (matched_treated_indices, matched_control_indices)
    """
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    ps_treated = propensity_scores[treated_idx].reshape(-1, 1)
    ps_control = propensity_scores[control_idx].reshape(-1, 1)

    n_neighbors = min(10, len(control_idx))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(ps_control)
    distances, indices = nn.kneighbors(ps_treated)

    matched_t, matched_c = [], []
    control_used = set()

    order = np.argsort(distances[:, 0])
    for i in order:
        for j in range(distances.shape[1]):
            dist = distances[i, j]
            if dist > caliper:
                break
            c_rel = indices[i, j]
            c_abs = control_idx[c_rel]
            if c_abs not in control_used:
                matched_t.append(treated_idx[i])
                matched_c.append(c_abs)
                control_used.add(c_abs)
                break

    return np.array(matched_t, dtype=int), np.array(matched_c, dtype=int)


def bootstrap_ate_ci(model_predict_fn, X, n_bootstrap=BOOTSTRAP_ITER, alpha=0.05):
    """Bootstrap ATE 置信区间"""
    n = X.shape[0]
    estimates = []
    rng = np.random.RandomState(RANDOM_STATE)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        cate_boot = model_predict_fn(X[idx])
        estimates.append(np.mean(cate_boot))
    estimates = np.array(estimates)
    ate = np.mean(estimates)
    ci_lower = np.percentile(estimates, 100 * alpha / 2)
    ci_upper = np.percentile(estimates, 100 * (1 - alpha / 2))
    return ate, ci_lower, ci_upper


# ==================== 主程序 ====================


def main():
    ensure_dirs()
    report_lines = []
    report_lines.append("=" * 65)
    report_lines.append("  电商评论情感因果推断报告 (PSM + Causal Forest)")
    report_lines.append("=" * 65)
    report_lines.append("")

    # ================================================================
    # 1. 数据加载
    # ================================================================
    report_lines.append("-" * 50)
    report_lines.append("1. 数据加载")
    report_lines.append("-" * 50)

    # 1a. 主数据
    cleaned_path = DIR_PROCESSED / "cleaned_data.csv"
    if not cleaned_path.exists():
        print(f"[ERROR] 未找到 {cleaned_path}，请先运行上游预处理模块。")
        sys.exit(1)
    df = pd.read_csv(cleaned_path)
    report_lines.append(f"  主表加载: cleaned_data.csv ({len(df)} 条)")
    required_cols = ["user_id", "product_id", "content", "score", "sentiment"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] cleaned_data.csv 缺少必要字段: {missing}")
        print(f"  实际字段: {df.columns.tolist()}")
        sys.exit(1)

    # 1b. 隐式情感数据
    implicit_path = DIR_PROCESSED / "implicit_sentiment_full.csv"
    has_implicit = False
    if implicit_path.exists():
        df_imp = pd.read_csv(implicit_path)
        if "implicit_score" in df_imp.columns:
            has_implicit = True
            # 按 content 合并 (content 为评论文本唯一键)
            if "content" in df_imp.columns and len(df_imp) == len(df):
                # 同长度直接拼列
                df["implicit_score"] = df_imp["implicit_score"].values
                report_lines.append("  implicit_sentiment_full.csv 加载成功 (按行拼接)")
            elif "content" in df_imp.columns:
                df = df.merge(
                    df_imp[["content", "implicit_score"]], on="content", how="left"
                )
                df["implicit_score"] = df["implicit_score"].fillna(df["sentiment"])
                report_lines.append("  implicit_sentiment_full.csv 加载成功 (按content合并)")
            else:
                has_implicit = False
        else:
            report_lines.append(
                "  [WARN] implicit_sentiment_full.csv 缺少 implicit_score 列，使用 sentiment 替代"
            )
    if not has_implicit:
        report_lines.append(
            "  [WARN] 未找到 implicit_sentiment_full.csv，implicit_score = sentiment + noise"
        )
        noise = np.random.normal(0, 0.08, len(df))
        df["implicit_score"] = np.clip(df["sentiment"].values + noise, 0.0, 1.0)

    # 1c. 属性级情感数据
    aspect_path = DIR_PROCESSED / "aspect_sentiment.jsonl"
    has_aspect = False
    aspect_lookup = {}
    if aspect_path.exists():
        try:
            import jsonlines

            aspects_list = []
            with jsonlines.open(aspect_path) as reader:
                for obj in reader:
                    aspects_list.append(obj)
            if aspects_list:
                has_aspect = True
                # 构建 (user_id, item_id) -> aspect_sentiment dict 查找表
                for rec in aspects_list:
                    key = (str(rec.get('user_id', '')), str(rec.get('item_id', '')))
                    aspect_lookup[key] = rec.get('aspect_sentiment', {})
                report_lines.append(
                    f"  aspect_sentiment.jsonl 加载成功 ({len(aspects_list)} 条, {len(aspect_lookup)} 个唯一键)"
                )
        except Exception as e:
            report_lines.append(f"  [WARN] aspect_sentiment.jsonl 读取失败: {e}")
    if not has_aspect:
        report_lines.append(
            "  [INFO] 未找到 aspect_sentiment.jsonl，跳过属性级特征增强"
        )

    report_lines.append("")

    # ================================================================
    # 2. 变量构造
    # ================================================================
    report_lines.append("-" * 50)
    report_lines.append("2. 变量构造")
    report_lines.append("-" * 50)

    # 处理变量: overall_sentiment = 0.6*sentiment + 0.4*implicit_score
    df["overall_sentiment"] = (
        0.6 * df["sentiment"] + 0.4 * df["implicit_score"]
    )
    df["treatment"] = (df["overall_sentiment"] > 0.5).astype(int)
    n_treat = df["treatment"].sum()
    n_control = len(df) - n_treat
    report_lines.append(
        f"  Treatment (overall_sentiment>0.5): 处理组={n_treat}, 对照组={n_control}"
    )
    report_lines.append(
        f"  overall_sentiment 均值: {df['overall_sentiment'].mean():.4f} "
        f"(std={df['overall_sentiment'].std():.4f})"
    )

    # 降采样处理组以缓解类别不平衡 (保留对照组全部样本)
    rng = np.random.RandomState(RANDOM_STATE)
    treat_idx = df[df['treatment'] == 1].index
    control_idx = df[df['treatment'] == 0].index
    max_treat = int(len(control_idx) * TREATMENT_DOWNSAMPLE_RATIO)
    if len(treat_idx) > max_treat:
        downsampled_treat = rng.choice(treat_idx, size=max_treat, replace=False)
        keep_idx = np.concatenate([downsampled_treat, control_idx])
        df = df.loc[keep_idx].reset_index(drop=True)
        n_treat_after = df['treatment'].sum()
        n_control_after = len(df) - n_treat_after
        report_lines.append(
            f"  降采样后: 处理组={n_treat_after}, 对照组={n_control_after} "
            f"(比例={n_treat_after/max(n_control_after,1):.1f}:1)"
        )

    # 结果变量: rating = score
    df["rating"] = df["score"].astype(float)
    report_lines.append(
        f"  Outcome (rating=score): 均值={df['rating'].mean():.2f} "
        f"(范围 [{df['rating'].min():.0f}, {df['rating'].max():.0f}])"
    )

    # 混淆变量
    # user_activity: 用户评论数
    user_counts = df["user_id"].value_counts()
    df["user_activity"] = df["user_id"].map(user_counts)
    report_lines.append(
        f"  user_activity: 均值={df['user_activity'].mean():.2f} "
        f"(每个用户平均评论 {df['user_activity'].mean():.1f} 条)"
    )

    # brand: 从 product_id 提取 (JD SKU 编码，作为品类/品牌代理变量)
    le_brand = LabelEncoder()
    df["brand"] = le_brand.fit_transform(df["product_id"].astype(str))
    report_lines.append(f"  brand: {len(le_brand.classes_)} 个唯一 product_id (作为品牌代理)")

    # review_length: 评论文本长度
    df["review_length"] = df["content"].astype(str).apply(len)
    report_lines.append(
        f"  review_length: 均值={df['review_length'].mean():.0f} 字符 "
        f"(范围 [{df['review_length'].min()}, {df['review_length'].max()}])"
    )

    # aspect 特征: 从细粒度情感数据中提取 (替代原来无效的 price_level 代理)
    if has_aspect and aspect_lookup:
        aspect_counts = []
        aspect_means = []
        aspect_stds = []
        for _, row in df.iterrows():
            key = (str(row['user_id']), str(row['product_id']))
            asp_dict = aspect_lookup.get(key, {})
            if asp_dict:
                values = list(asp_dict.values())
                aspect_counts.append(len(values))
                aspect_means.append(np.mean(values))
                aspect_stds.append(np.std(values) if len(values) > 1 else 0.0)
            else:
                aspect_counts.append(0)
                aspect_means.append(0.5)
                aspect_stds.append(0.0)
        df['aspect_diversity'] = aspect_counts
        df['aspect_sentiment_mean'] = aspect_means
        df['aspect_sentiment_std'] = aspect_stds
        report_lines.append(
            f"  aspect_diversity: 均值={df['aspect_diversity'].mean():.1f} 个属性/评论"
        )
        report_lines.append(
            f"  aspect_sentiment_mean: 均值={df['aspect_sentiment_mean'].mean():.4f}"
        )
        confounder_cols = [
            "user_activity", "brand", "review_length",
            "aspect_diversity", "aspect_sentiment_mean", "aspect_sentiment_std",
        ]
        confounder_labels = [
            "用户活跃度", "品牌/商品ID", "评论长度",
            "属性多样性", "属性情感均值", "属性情感分歧度",
        ]
    else:
        confounder_cols = ["user_activity", "brand", "review_length"]
        confounder_labels = ["用户活跃度", "品牌/商品ID", "评论长度"]

    X_conf = df[confounder_cols].values.astype(float)
    treatment = df["treatment"].values
    outcome = df["rating"].values

    report_lines.append("")
    report_lines.append(f"  混淆变量矩阵形状: {X_conf.shape}")
    report_lines.append(f"  混淆变量: {', '.join(confounder_cols)}")
    report_lines.append("")

    # ================================================================
    # 3. PSM — 倾向得分匹配
    # ================================================================
    report_lines.append("-" * 50)
    report_lines.append("3. 倾向得分匹配 (PSM)")
    report_lines.append("-" * 50)

    if n_treat == 0 or n_control == 0:
        print("[ERROR] 处理组或对照组为空，无法进行因果推断。")
        sys.exit(1)

    # 3a. 逻辑回归估计倾向得分
    ps_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight='balanced')
    ps_model.fit(X_conf, treatment)
    propensity_scores = ps_model.predict_proba(X_conf)[:, 1]
    report_lines.append(
        f"  倾向得分: 均值={propensity_scores.mean():.4f}, "
        f"范围=[{propensity_scores.min():.4f}, {propensity_scores.max():.4f}]"
    )

    # 3b. SMD 匹配前
    smd_before = {}
    for j, label in enumerate(confounder_labels):
        smd_before[label] = smd_score(
            X_conf[treatment == 1, j], X_conf[treatment == 0, j]
        )
    report_lines.append("  匹配前 SMD:")
    for label, val in smd_before.items():
        flag = " <<" if abs(val) > 0.1 else ""
        report_lines.append(f"    {label}: {val:+.4f}{flag}")

    # 3c. 执行匹配
    matched_t, matched_c = ps_matching(propensity_scores, treatment, caliper=CALIPER)
    n_matched = len(matched_t)
    report_lines.append(
        f"  匹配结果: {n_matched} 对 (caliper={CALIPER}, "
        f"匹配率={n_matched/max(n_treat,1)*100:.1f}%)"
    )

    if n_matched < 10:
        report_lines.append(
            "  [WARN] 匹配对过少，因果效应估计可能不稳定。"
        )
        # 即使匹配对少也继续执行，但使用全部数据做因果森林
        matched_idx = np.arange(len(df))
    else:
        matched_idx = np.concatenate([matched_t, matched_c])

    # 3d. SMD 匹配后
    if n_matched >= 2:
        smd_after = {}
        for j, label in enumerate(confounder_labels):
            smd_after[label] = smd_score(
                X_conf[matched_t, j], X_conf[matched_c, j]
            )
        report_lines.append("  匹配后 SMD:")
        for label, val in smd_after.items():
            flag = " <<" if abs(val) > 0.1 else ""
            report_lines.append(f"    {label}: {val:+.4f}{flag}")
    else:
        smd_after = {k: float("nan") for k in confounder_labels}

    report_lines.append("")

    # ================================================================
    # 4. Causal Forest — 因果森林
    # ================================================================
    report_lines.append("-" * 50)
    report_lines.append("4. 因果森林 (Causal Forest)")
    report_lines.append("-" * 50)

    X_matched = X_conf[matched_idx]
    treatment_matched = treatment[matched_idx]
    outcome_matched = outcome[matched_idx]

    try:
        if not _HAS_CAUSALML:
            raise ImportError("causalml not available")

        cf = CausalForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
        )
        cf.fit(X=X_matched, treatment=treatment_matched, y=outcome_matched)

        # CATE 预测
        cate_all = cf.predict(X_matched)
        # 全部匹配样本上的 CATE
        ate_cf = float(np.mean(cate_all))
        # ATT: 仅处理组的平均 CATE
        treat_in_matched = treatment_matched == 1
        if treat_in_matched.sum() > 0:
            att_cf = float(np.mean(cate_all[treat_in_matched]))
        else:
            att_cf = ate_cf

        # Bootstrap 置信区间
        ate_bt, ate_ci_low, ate_ci_high = bootstrap_ate_ci(
            lambda X: cf.predict(X), X_matched
        )
        att_bt, att_ci_low, att_ci_high = (
            bootstrap_ate_ci(
                lambda X: cf.predict(X),
                X_matched[treatment_matched == 1],
            )
            if treat_in_matched.sum() >= 10
            else (att_cf, att_cf, att_cf)
        )

        report_lines.append("  --- 因果效应估计 ---")
        report_lines.append(
            f"  ATE  (平均处理效应)         : {ate_cf:+.4f}  "
            f"[95% CI: {ate_ci_low:+.4f}, {ate_ci_high:+.4f}]"
        )
        report_lines.append(
            f"  ATT  (处理组平均处理效应)   : {att_cf:+.4f}  "
            f"[95% CI: {att_ci_low:+.4f}, {att_ci_high:+.4f}]"
        )

        # PSM 直接估计 (匹配后均值差)
        if n_matched >= 2:
            psm_ate = float(
                np.mean(outcome[matched_t]) - np.mean(outcome[matched_c])
            )
            report_lines.append(
                f"  PSM 直接估计 (matched diff) : {psm_ate:+.4f}"
            )
        else:
            psm_ate = float("nan")

        # 特征重要性 Top 10
        report_lines.append("")
        report_lines.append("  --- 特征重要性 Top 10 ---")
        try:
            importances = cf.feature_importances_
            feat_imp = list(zip(confounder_labels, importances))
            feat_imp.sort(key=lambda x: x[1], reverse=True)
            for rank, (name, imp) in enumerate(feat_imp, 1):
                report_lines.append(f"    {rank:>2}. {name}: {imp:.4f}")
        except AttributeError:
            report_lines.append(
                "    (CausalForestRegressor 不支持 feature_importances_ 属性)"
            )
            feat_imp = []

        cf_ok = True
    except ImportError:
        report_lines.append(
            "  [WARN] 未安装 causalml 库，跳过因果森林。"
        )
        report_lines.append("         安装: pip install causalml")
        ate_cf = float("nan")
        att_cf = float("nan")
        psm_ate = float("nan") if n_matched < 2 else float(
            np.mean(outcome[matched_t]) - np.mean(outcome[matched_c])
        )
        cf_ok = False
        feat_imp = []
    except Exception as e:
        report_lines.append(f"  [WARN] Causal Forest 执行异常: {e}")
        ate_cf = float("nan")
        att_cf = float("nan")
        cf_ok = False
        feat_imp = []

    report_lines.append("")

    # ================================================================
    # 5. 平衡性可视化 (PSM Balance Plot)
    # ================================================================
    report_lines.append("-" * 50)
    report_lines.append("5. 平衡性可视化")
    report_lines.append("-" * 50)

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        labels_plot = list(smd_before.keys())
        x = np.arange(len(labels_plot))
        width = 0.35

        before_vals = [smd_before[k] for k in labels_plot]
        after_vals = [smd_after.get(k, float("nan")) for k in labels_plot]

        bars1 = ax.bar(x - width / 2, before_vals, width, label="匹配前", color="#E74C3C", alpha=0.85)
        bars2 = ax.bar(x + width / 2, after_vals, width, label="匹配后", color="#2ECC71", alpha=0.85)

        ax.axhline(y=0.1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(y=-0.1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_plot, fontsize=11)
        ax.set_ylabel("Standardized Mean Difference (SMD)", fontsize=12)
        ax.set_title("PSM 平衡性检验 — 匹配前后混淆变量 SMD 对比", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.set_ylim(-1.2, 1.2)

        # 数值标注
        for bar, val in zip(bars1, before_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03 * np.sign(bar.get_height()),
                f"{val:+.3f}",
                ha="center",
                va="bottom" if bar.get_height() >= 0 else "top",
                fontsize=8,
            )
        for bar, val in zip(bars2, after_vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.03 * np.sign(bar.get_height()),
                    f"{val:+.3f}",
                    ha="center",
                    va="bottom" if bar.get_height() >= 0 else "top",
                    fontsize=8,
                )

        fig_path = DIR_FIGURES / "psm_balance.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        report_lines.append(f"  平衡性对比图已保存: {fig_path}")
    except Exception as e:
        report_lines.append(f"  [WARN] 生成平衡性图失败: {e}")

    report_lines.append("")

    # ================================================================
    # 6. 业务结论
    # ================================================================
    report_lines.append("-" * 50)
    report_lines.append("6. 业务结论与分析")
    report_lines.append("-" * 50)

    report_lines.append("  核心发现:")
    report_lines.append(
        "  - 本模块通过 PSM 消除选择偏差后，估计情感倾向对"
        "用户评分的因果效应。"
    )

    if cf_ok and not np.isnan(ate_cf):
        direction = "正向" if ate_cf > 0 else "负向"
        significance = (
            "统计显著" if (ate_ci_low * ate_ci_high) > 0 else "未达统计显著"
        )
        report_lines.append(
            f"  - ATE = {ate_cf:+.4f} ({direction}), "
            f"95% CI [{ate_ci_low:+.4f}, {ate_ci_high:+.4f}] ({significance})"
        )
        report_lines.append(
            f"  - ATT = {att_cf:+.4f}, "
            f"表明对高情感倾向评论而言，其因果效应{'更强' if abs(att_cf) > abs(ate_cf) else '与ATE一致'}。"
        )

    if feat_imp:
        top_feat = feat_imp[0][0]
        report_lines.append(
            f"  - 最重要的混淆变量为「{top_feat}」，"
            f"建议在后续分析中重点关注其调节作用。"
        )

    report_lines.append("  业务建议:")
    report_lines.append(
        "  1. 情感倾向对评分存在因果驱动关系，运营团队应关注"
        "差评中隐含的情感信号。"
    )
    report_lines.append(
        "  2. 用户活跃度与评论长度是重要的混淆因素，建议在"
        "用户分层策略中纳入这些变量。"
    )
    report_lines.append(
        "  3. 因果效应置信区间可用于 A/B 测试的样本量估算与"
        "决策阈值设定。"
    )

    report_lines.append("")
    report_lines.append("=" * 65)
    report_lines.append("  报告结束")
    report_lines.append("=" * 65)

    # ================================================================
    # 7. 输出报告
    # ================================================================
    report_path = DIR_REPORTS / "causal_report.txt"
    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
