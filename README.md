# 本地市场风险引擎（Python）

对本地多组持仓与历史行情计算**历史模拟风险价值（VaR）与预期缺口（ES）**，
支持滚动窗口 / 指数衰减加权、增量更新、并发查询、取消超时与崩溃恢复。

- 计算口径与配置项：见 [`docs/risk-methodology.md`](docs/risk-methodology.md)
- 测试：`uv run pytest`（66+ 用例，覆盖滚动窗口、增量与全量一致性、异常
  输入、并发取消、崩溃恢复与性能边界）

## 快速开始

```bash
uv sync --group dev
uv run uvicorn main:app   # 或 uv run uvicorn market_risk.api:create_app
```

环境变量：`RISK_STORE`（数据目录，默认 ./.risk-data）、
`RISK_MAX_CONCURRENCY`（默认 4）。

### HTTP 流程

```bash
# 1) 提交行情批次（乱序可自动排序；缺失/非正/重复/迟到整批拒绝）
curl -X POST localhost:8000/api/v1/history -H 'content-type: application/json' -d '{
  "as_of": "2025-01-31",
  "observations": [
    {"instrument": "AAPL", "day": "2025-01-02", "price": 101.2},
    {"instrument": "AAPL", "day": "2025-01-01", "price": 100.0}
  ]
}'
# -> {"data_version": 1, "accepted": 2}

# 2) 注册组合
curl -X POST localhost:8000/api/v1/portfolios -H 'content-type: application/json' -d '{
  "portfolio_id": "fund-a",
  "positions": [{"instrument": "AAPL", "quantity": 100}]}'

# 3) 提交计算（等权滚动 / 衰减加权二选一）
curl -X POST localhost:8000/api/v1/calculations -H 'content-type: application/json' -d '{
  "portfolio_id": "fund-a",
  "config": {"confidence": 0.99, "horizon": 1, "method": "ROLLING", "window": 250}}'
# -> {"task_id": "task:fund-a:1:c=0.99:h=1:r:250"}

# 4) 读取结果 / 状态 / 取消
curl "localhost:8000/api/v1/results/<task_id>?timeout=30"
curl "localhost:8000/api/v1/tasks/<task_id>"
curl -X POST "localhost:8000/api/v1/tasks/<task_id>/cancel"

# 5) 崩溃后恢复（重启进程后调用，幂等）
curl -X POST localhost:8000/api/v1/recover
```

衰减加权配置：`"method": "DECAYED", "decay_lambda": 0.97`。

### 直接使用 Python API

```python
from datetime import date
from market_risk.engine import RiskEngine
from market_risk.models import (HistoryBatch, Portfolio, Position,
                                PriceObservation, RiskConfig)

with RiskEngine("./.risk-data", max_concurrency=4) as eng:
    eng.submit_history(HistoryBatch(
        (PriceObservation("A", date(2025, 1, 1), 100.0),
         PriceObservation("A", date(2025, 1, 2), 99.0)),
        as_of=date(2025, 1, 2)))
    eng.register_portfolio(Portfolio("a", (Position("A", 10.0),)))
    tid = eng.calculate("a", RiskConfig(confidence=0.95, window=2))
    print(eng.get_result(tid).var, eng.get_result(tid).es)
```

## 工程结构

```
src/market_risk/
  models.py      # 领域类型、错误码、配置
  validation.py  # 缺失/非正/重复/乱序/迟到校验（全有或全无）
  series.py      # 有序价格序列、有界淘汰、多标的窗口对齐
  simulation.py  # 情景损益、等权/衰减权重、VaR/ES、持有期缩放
  storage.py     # WAL + 原子结果文件（崩溃恢复）
  tasks.py       # 有界线程池、取消令牌、状态机
  engine.py      # 门面：版本隔离、增量重算、恢复
  api.py         # FastAPI 接口
```
