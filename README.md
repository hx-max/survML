# survML

> **survML** 是基于 [Microsoft FLAML](https://github.com/microsoft/FLAML) 的二次开发项目：为 FLAML 新增了一个 **生存分析（Survival Analysis）任务类型**，让 `AutoML.fit()` 也能一键完成生存模型的选择、超参调优与 C-index 评估。

```python
from flaml import AutoML

automl = AutoML()
automl.fit(X, y, task="survival", metric="c_index", time_budget=15)
print("best C-index:", 1 - automl.best_loss)
```

`y` 是**右删失（right-censored）**目标，两列 `(duration, event)`：
- `duration`：生存时间（如用户的存活时长、设备到故障的时间）
- `event`：是否发生终点事件（1 = 事件发生，0 = 被删失）

---

## 为什么做这个

FLAML 原生支持分类、回归、排序、时序预测、NLP，但**没有生存分析**。它只有一个 `SparkAFTSurvivalRegressionEstimator`，还被当成普通回归在用（`assert self._task.is_regression()`），既没有 C-index 指标，也没有右删失数据管线。

survML 填补了这个空白：新增 `survival` 任务，打通了「任务类型 → 数据校验 → 切分 → 模型 → 指标」全链路。

---

## 快速开始

```bash
git clone https://github.com/hx-max/survML.git
cd survML
pip install -e ".[automl,test]"     # FLAML 依赖（xgboost / lightgbm 等）
pip install lifelines               # 生存分析依赖
python survival_demo.py
```

> 注：survML 的改动尚未发布到 PyPI，体验 `task="survival"` 需按上面的方式克隆安装。

---

## 我改了什么

共 **5 个文件，+121 / -10 行**：

| 文件 | 改动 |
|---|---|
| `flaml/automl/task/task.py` | 新增 `SURVIVAL` 任务名常量 + `is_survival()` 判断方法 |
| `flaml/automl/model.py` | 新增 `CoxPHEstimator`（封装 `lifelines.CoxPHFitter`）；`_estimator_type` 支持 survival 三态 |
| `flaml/automl/task/generic_task.py` | 注册 `"coxph"` 估计器；`default_metric` 返回 `c_index`；打通右删失数据管线（校验/切分/交叉验证）；survival 只搜 `coxph` |
| `flaml/automl/ml.py` | 新增 `c_index` 指标（concordance index，转成 loss） |
| `flaml/automl/automl.py` | `AutoML` 层的 `_estimator_type` 支持 survival |

---

## 技术要点

1. **右删失数据表示**：生存目标 `y` 是 `(duration, event)` 两列。FLAML 原数据管线假设 `y` 是一维标签，survML 在 `validate_data` / `evaluate_model_CV` 中对 DataFrame 型 `y` 用 `.iloc` 按行切分、跳过标签编码器。
2. **指标方向**：C-index 是「越大越好」，而 FLAML 内部统一用「越小越好」的 loss，因此 `c_index` 指标返回 `1 - concordance_index`。
3. **风险分符号约定**：`lifelines.concordance_index` 约定「分数越高 = 生存越久（风险越低）」，而 `predict_partial_hazard` 输出「越高 = 风险越高」，因此指标内部对风险分取负。
4. **任务类型三态化**：sklearn 的 estimator 只有 classifier/regressor 两态，survML 把 `_estimator_type` 扩展为 classifier/survival/regressor 三态。

---

## 已知限制 / 下一步

- 目前只接入 `CoxPH`（lifelines）。下一步可加 `sksurv` 的随机生存森林、梯度提升生存模型，让 AutoML 在多个生存模型间搜索。
- 尚未实现 `predict()` 输出完整生存函数曲线（当前输出风险分），可接入 Kaplan-Meier / 生存曲线绘制。
- 交叉验证默认使用 `uniform`（K-Fold），未做按事件的删失感知分层。

---

## 许可

本项目是 [Microsoft FLAML](https://github.com/microsoft/FLAML)（MIT License）的派生作品，保留原始版权声明（见 [NOTICE.md](NOTICE.md)）。
