"""FastAPI HTTP 接口：批次提交、组合注册、计算、状态查询、结果、取消。"""

from __future__ import annotations

import math
import os
from datetime import date

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from market_risk.models import (
    DEFAULT_MAX_CONCURRENCY,
    ErrorCode,
    HistoryBatch,
    Portfolio,
    Position,
    PriceObservation,
    RiskConfig,
    RiskEngineError,
    SimulationMethod,
)
from market_risk.engine import RiskEngine


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------
class ObservationIn(BaseModel):
    instrument: str
    day: date
    price: float

    @field_validator("price")
    @classmethod
    def _price_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("价格必须为有限数值（缺失/NaN/Inf 被拒绝）")
        return v


class HistoryBatchIn(BaseModel):
    observations: list[ObservationIn] = Field(min_length=1)
    as_of: date | None = None


class PositionIn(BaseModel):
    instrument: str
    quantity: float

    @field_validator("quantity")
    @classmethod
    def _qty_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("持仓数量必须为有限数值")
        return v


class PortfolioIn(BaseModel):
    portfolio_id: str
    positions: list[PositionIn] = Field(min_length=1)


class RiskConfigIn(BaseModel):
    confidence: float = 0.99
    horizon: int = 1
    method: SimulationMethod = SimulationMethod.ROLLING
    window: int = 250
    decay_lambda: float = 0.99


class CalculateIn(BaseModel):
    portfolio_id: str
    config: RiskConfigIn = Field(default_factory=RiskConfigIn)
    data_version: int | None = None


# ---------------------------------------------------------------------------
def create_app(
    engine: RiskEngine | None = None,
    *,
    store_path: str | None = None,
    max_concurrency: int | None = None,
) -> FastAPI:
    """构造 FastAPI 应用；engine 可注入（测试用），否则用全局环境配置。"""
    if engine is None:
        engine = RiskEngine(
            store_path or os.environ.get("RISK_STORE", "./.risk-data"),
            max_concurrency=max_concurrency
            or int(os.environ.get("RISK_MAX_CONCURRENCY",
                                  DEFAULT_MAX_CONCURRENCY)),
        )
    app = FastAPI(title="Local Market Risk Engine", version="0.1.0")
    app.state.engine = engine
    router = APIRouter()

    @router.get("/health")
    def health() -> dict:
        return {"status": "ok", "data_version": engine.store.version}

    @router.post("/history")
    def submit_history(body: HistoryBatchIn) -> dict:
        batch = HistoryBatch(
            observations=tuple(
                PriceObservation(o.instrument, o.day, o.price)
                for o in body.observations),
            as_of=body.as_of,
        )
        version = engine.submit_history(batch)
        return {"data_version": version, "accepted": len(batch.observations)}

    @router.post("/portfolios")
    def register_portfolio(body: PortfolioIn) -> dict:
        portfolio = Portfolio(
            portfolio_id=body.portfolio_id,
            positions=tuple(
                Position(p.instrument, p.quantity) for p in body.positions),
        )
        engine.register_portfolio(portfolio)
        return {"portfolio_id": body.portfolio_id, "registered": True}

    @router.post("/calculations")
    def calculate(body: CalculateIn) -> dict:
        config = RiskConfig(
            confidence=body.config.confidence,
            horizon=body.config.horizon,
            method=body.config.method,
            window=body.config.window,
            decay_lambda=body.config.decay_lambda,
        )
        task_id = engine.calculate(
            body.portfolio_id, config, data_version=body.data_version)
        return {"task_id": task_id}

    @router.get("/tasks/{task_id}")
    def task_status(task_id: str) -> dict:
        st = engine.get_task_status(task_id)
        return {
            "task_id": st.task_id,
            "state": st.state.value,
            "portfolio_id": st.portfolio_id,
            "error_code": st.error_code,
            "error_reason": st.error_reason,
        }

    @router.get("/results/{task_id}")
    def get_result(task_id: str, timeout: float | None = None) -> dict:
        r = engine.get_result(task_id, timeout=timeout)
        return {
            "task_id": task_id,
            "portfolio_id": r.portfolio_id,
            "data_version": r.data_version,
            "config_version": r.config_version,
            "var": r.var,
            "es": r.es,
            "sample_size": r.sample_size,
            "as_of": r.as_of.isoformat(),
            "method": r.method.value,
        }

    @router.post("/tasks/{task_id}/cancel")
    def cancel(task_id: str) -> dict:
        return {"task_id": task_id, "cancelled": engine.cancel_task(task_id)}

    @router.post("/tasks/{task_id}/recalculate")
    def recalculate(task_id: str) -> dict:
        return {"task_id": engine.recalculate(task_id)}

    @router.post("/recover")
    def recover() -> dict:
        return {"requeued": engine.recover()}

    @app.exception_handler(RiskEngineError)
    async def risk_error_handler(request, exc):  # noqa: ANN001
        return JSONResponse(
            status_code=400,
            content={"error_code": exc.code.value, "reason": exc.reason},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request,
                                 exc: RequestValidationError):
        # 区分：空批次/缺字段/NaN 等 -> MISSING_PRICE 或 INVALID_PARAMETER
        errs = exc.errors()
        code = ErrorCode.INVALID_PARAMETER.value
        reasons = []
        for e in errs:
            loc = ".".join(str(x) for x in e.get("loc", ()))
            reasons.append(f"{loc}: {e.get('msg')}")
            if "price" in loc or "observations" in loc:
                code = ErrorCode.MISSING_PRICE.value
        return JSONResponse(
            status_code=400,
            content={"error_code": code, "reason": "; ".join(reasons)},
        )

    @app.exception_handler(KeyError)
    async def key_error_handler(request, exc):  # noqa: ANN001
        return JSONResponse(
            status_code=404,
            content={"error_code": "TASK_NOT_FOUND",
                     "reason": str(exc).strip("'")},
        )

    app.include_router(router, prefix="/api/v1")
    return app
