"""HTTP 入口：uvicorn main:app"""

from market_risk.api import create_app

app = create_app()
