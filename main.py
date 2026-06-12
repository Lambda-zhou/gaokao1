import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.models import ConsultRequest
from middleware.error_handler import (
    app_exception_handler,
    global_exception_handler,
    AppException,
)
from middleware.logging import setup_logging, logging_middleware
from middleware.rate_limit import rate_limit_middleware
from api import consult, evaluate, data, agent, sessions, llm

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_ENTRY = BASE_DIR / "zhiyuan-agent.html"
ASSETS_DIR = BASE_DIR / "assets"
IMAGES_DIR = BASE_DIR / "images"

# 配置日志
setup_logging()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="张雪峰思维操作系统后端 API。提供基于5个核心心智模型和8条决策启发式的智能咨询、决策评估与Agent服务。",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# 注册异常处理器
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# 注册中间件
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(logging_middleware)

# CORS：从配置读取，不再硬编码 ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Routers
app.include_router(consult.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")
app.include_router(llm.router, prefix="/api")

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


@app.get("/zhiyuan-agent.html", include_in_schema=False)
async def zhiyuan_agent_page():
    return FileResponse(FRONTEND_ENTRY)


@app.get("/app", include_in_schema=False)
async def app_page():
    return FileResponse(FRONTEND_ENTRY)


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs" if settings.debug else None,
        "endpoints": {
            "consult": "POST /api/consult (支持 session_id 多轮对话)",
            "sessions": {
                "create": "POST /api/sessions",
                "list": "GET /api/sessions",
                "get": "GET /api/sessions/{id}",
                "delete": "DELETE /api/sessions/{id}",
                "update_profile": "PUT /api/sessions/{id}/profile",
                "rename": "PUT /api/sessions/{id}/rename",
            },
            "llm": {
                "test": "POST /api/llm/test (测试用户填写的 OpenAI Chat 兼容配置)",
            },
        },
    }


@app.get("/health")
async def health():
    from core.llm_client import llm_client
    from core.config import settings
    return {
        "status": "ok",
        "llm_available": llm_client.is_available(),
        "llm_provider": settings.llm_provider,
        "llm_model": llm_client.model,
        "llm_provider_label": llm_client.provider_label,
        "llm_base_url": llm_client.openai_base_url,
        "llm_last_error_type": llm_client.last_error_type,
        "llm_last_error_summary": llm_client.last_error_summary,
        "recognition_version": "school_chance_alias_v2",
    }


@app.post("/debug/intent")
async def debug_intent(request: ConsultRequest):
    from core.consult_orchestrator import consult_orchestrator

    enriched = consult_orchestrator._enrich_request_context(request)
    intent = consult_orchestrator._detect_intent(enriched)
    queries = consult_orchestrator._build_research_queries(enriched, intent)
    return {
        "question": enriched.question,
        "context": enriched.context.model_dump(exclude_none=True) if enriched.context else None,
        "intent": intent.intent,
        "schools": intent.school_names,
        "majors": intent.major_names,
        "needs_research": intent.needs_research,
        "queries": queries[:10],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=settings.debug)
