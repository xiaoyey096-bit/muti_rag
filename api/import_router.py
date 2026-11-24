import os
import os.path
import time

import uvicorn
from fastapi import FastAPI, UploadFile, Depends, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from muti_rag.core.paths import get_front_page_dir
from muti_rag.schema.upload_schema import (
    UploadResponse, TaskStatusResponse, KbCreateRequest, KbUpdateRequest, KbResponse,
    UploadInitRequest, TaskCreateRequest,
)
from muti_rag.service.upload_service import UpLoadService
from muti_rag.service.task_queue import get_task_queue
from muti_rag.core.deps import get_upload_file_service
from muti_rag.utils.task_util import (
    get_task_info, get_task_doc, request_resume,
    TASK_STATUS_QUEUED, TASK_STATUS_PROCESSING, TASK_STATUS_WAITING_RETRY,
)
from muti_rag.utils import upload_session_util as uploads
from muti_rag.utils.resilience import RESILIENCE
from muti_rag.utils.env_selfcheck import ensure_pypdf_stream_limit
from muti_rag.utils import kb_registry_util as kb_registry


# 1. 创建fastapi实例
# 2. 注册路由（将上传请求以及查询导入任务的请求注册到fastapi实例上）
# 3. 利用uvicorn服务器启动fastapi


def create_app():
    """
    创建FastAPI实例
    Returns:
    FastAPI实例

    """

    # 0. 启动自检与恢复：pypdf 流长度上限 + 上一次进程遗留任务重新排队
    ensure_pypdf_stream_limit()
    get_task_queue().start()

    # 1. 实例化
    app = FastAPI(description="格物智答导入的应用", version="v1.2")

    # 2. 跨域配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # ← 和 credentials=True 冲突
        allow_credentials=False,  # 自定义cookies Authorization  tsl客户端证书信息
        allow_methods=["*"],  # ← 和 credentials=True 冲突  GET(获取资源)  POST(新增)  DELETE（删除）  PUT（修改）
        allow_headers=["*"],  # ← 和 credentials=True 冲突   自定义的头字段 token  content-type:application/json
    )

    # 3. 挂载静态文件（import.html）
    page_dir = get_front_page_dir()

    if page_dir and os.path.exists(page_dir):
        app.mount("/front", StaticFiles(directory=page_dir))

    # 4. 注册路由
    register_router(app)

    # 5. 返回fastapi实例
    return app


def register_router(app: FastAPI):
    @app.get("/")
    def index():
        return RedirectResponse(url="/front/index.html")

    @app.get("/health")
    def health():
        return {"flag": "success"}

    # ==================== 文件层：分片断点续传 ====================

    @app.post("/uploads/init")
    def uploads_init_endpoint(req: UploadInitRequest):
        """
        初始化上传：
        - 同 sha256 已有完整文件 → 秒传，直接返回 file_id；
        - 同 sha256 有未完成会话 → 返回已收分片，前端只补传缺失部分；
        - 否则新建会话。
        """
        free = uploads.free_bytes(uploads.UPLOAD_ROOT)
        need = max(int(req.size) * 2, uploads.MIN_FREE_MB * 1024 * 1024)
        if 0 <= free < need:
            raise HTTPException(status_code=507, detail="服务器磁盘空间不足，请稍后重试")
        return uploads.init_upload(req.filename, req.size, req.sha256,
                                   req.chunk_size, req.kb_id, req.product_name)

    @app.put("/uploads/{upload_id}/parts/{part_no}")
    async def uploads_put_part_endpoint(upload_id: str, part_no: int, request: Request):
        """上传单个分片（原始二进制 body，5MB/片）。同片重传覆盖，天然幂等。"""
        body = await request.body()
        try:
            return uploads.save_part(upload_id, part_no, body)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e.args[0] if e.args else e))
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/uploads/{upload_id}")
    def uploads_get_endpoint(upload_id: str):
        """查询会话（刷新页面后恢复上传用）。"""
        doc = uploads.get_upload(upload_id)
        if not doc:
            raise HTTPException(status_code=404, detail="上传会话不存在或已过期")
        from muti_rag.utils.upload_session_util import _session_public
        return dict(_session_public(doc), completed=doc.get("status") == uploads.S_COMPLETED,
                    resumed=True)

    @app.post("/uploads/{upload_id}/complete")
    def uploads_complete_endpoint(upload_id: str):
        """合并分片 + sha256 校验 + 归档。缺片/校验失败时返回可操作的 JSON。"""
        try:
            return uploads.complete_upload(upload_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e.args[0] if e.args else e))
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.post("/uploads/{upload_id}/abort")
    def uploads_abort_endpoint(upload_id: str):
        return uploads.abort_upload(upload_id)

    @app.post("/files/{file_id}/tasks")
    def create_task_endpoint(file_id: str, req: TaskCreateRequest):
        """以 file_id 创建导入任务（同文件+同知识库+同标题自动去重）。"""
        try:
            return uploads.create_task_for_file(file_id, req.kb_id, req.product_name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e.args[0] if e.args else e))

    # ==================== 旧单请求上传（兼容入口） ====================

    @app.post("/upload", response_model=UploadResponse)
    def upload_endpoint(file: UploadFile,
                        kb_id: str = Form(""),
                        product_name: str = Form(""),
                        upload_service: UpLoadService = Depends(get_upload_file_service)):
        """
        兼容旧前端：内部走"落库 + 登记 files + 建任务 + 入队"，
        与分片上传共用同一套任务队列。
        """
        task_id, import_file_path, file_dir, resolved_kb_id, file_sha256, file_id = \
            upload_service.process_upload_file(file, kb_id=kb_id, product_name=product_name)

        get_task_queue().enqueue(task_id)

        return UploadResponse(message=f"{file.filename}文件上传成功", task_id=task_id, kb_id=resolved_kb_id)

    # ==================== 任务状态 / 续跑 ====================

    @app.get("/status/{task_id}")
    def get_task_status_endpoint(task_id: str):
        """
        查询上传任务：前端 1.5s 轮询。
        waiting_retry 表示系统正在自动重试，前端继续保持轮询即可。
        """
        task_info = get_task_info(task_id)
        return TaskStatusResponse(**task_info)

    @app.post("/tasks/{task_id}/resume")
    def resume_task_endpoint(task_id: str):
        """
        断点续跑：校验任务与原始文件后，重新入队；队列从最后成功步骤继续。
        """
        doc = get_task_doc(task_id)
        if not doc:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")
        status = doc.get("status")
        if status in (TASK_STATUS_QUEUED, TASK_STATUS_PROCESSING):
            raise HTTPException(status_code=409, detail="任务仍在运行中，无需继续导入")
        if status == TASK_STATUS_WAITING_RETRY:
            raise HTTPException(status_code=409, detail="任务正在自动重试，请稍候")
        if not doc.get("import_file_path") or not os.path.exists(doc["import_file_path"]):
            raise HTTPException(status_code=410, detail="原始文件已不存在，请重新上传")

        request_resume(task_id)
        get_task_queue().enqueue(task_id)
        return {"task_id": task_id, "status": "queued"}

    # ==================== 运维观测 ====================

    @app.get("/ops/metrics")
    def ops_metrics_endpoint():
        """任务队列、步骤耗时/失败、熔断器、上传会话、磁盘的可观测快照。"""
        from muti_rag.utils.client.storage_clients import StorageClients
        metrics = {"queue": get_task_queue().status(), "disk_free_mb": round(uploads.free_bytes() / 1048576),
                   "breakers": RESILIENCE.snapshot()}
        try:
            db = StorageClients.get_mongo_db()
            counts = list(db["import_tasks"].aggregate(
                [{"$group": {"_id": "$status", "n": {"$sum": 1}}}]))
            metrics["tasks"] = {row["_id"] or "unknown": row["n"] for row in counts}
            metrics["uploads_open"] = db["upload_sessions"].count_documents({"status": "OPEN"})
            since = time.time() - 86400
            steps = list(db["import_task_steps"].aggregate([
                {"$match": {"finished_at": {"$gte": since}}},
                {"$group": {"_id": {"step": "$step", "status": "$status"},
                            "n": {"$sum": 1}, "avg_ms": {"$avg": "$duration_ms"}}},
            ]))
            metrics["steps_24h"] = [
                {"step": r["_id"]["step"], "status": r["_id"]["status"], "count": r["n"],
                 "avg_ms": int(r.get("avg_ms") or 0)} for r in steps
            ]
        except Exception as e:
            metrics["mongo_error"] = str(e)
        return metrics

    # ==================== 知识库管理 ====================
    @app.get("/kbs", response_model=list[KbResponse])
    def list_kbs_endpoint():
        """列出全部知识库（供导入页管理、查询页选择产品）。"""
        return [KbResponse(**kb) for kb in kb_registry.list_kbs()]

    @app.post("/kbs", response_model=KbResponse)
    def create_kb_endpoint(req: KbCreateRequest):
        """新建一个空知识库（先建产品身份，再往里上传文档）。"""
        kb_id = kb_registry.create_kb(req.product_name, req.display_name, req.hint)
        return KbResponse(**kb_registry.get_kb(kb_id))

    @app.patch("/kbs/{kb_id}", response_model=KbResponse)
    def update_kb_endpoint(kb_id: str, req: KbUpdateRequest):
        """更新知识库展示名/提示/产品名。"""
        if not kb_registry.get_kb(kb_id):
            raise HTTPException(status_code=404, detail="知识库不存在")
        kb_registry.update_kb_meta(kb_id, display_name=req.display_name,
                                   hint=req.hint, product_name=req.product_name)
        return KbResponse(**kb_registry.get_kb(kb_id))

    @app.delete("/kbs/{kb_id}")
    def delete_kb_endpoint(kb_id: str):
        """删除知识库：清理 Milvus 中该 kb_id 的 chunks 与商品名记录，再删注册表。"""
        if not kb_registry.get_kb(kb_id):
            raise HTTPException(status_code=404, detail="知识库不存在")

        from muti_rag.utils.client.storage_clients import StorageClients
        from muti_rag.processor.import_processor.config import get_config
        cfg = get_config()
        purged = {"chunks": 0, "item_names": 0}
        try:
            mc = StorageClients.get_milvus_client()
            expr = 'kb_id == "%s"' % kb_id.replace('"', '\\"')
            for key, coll in (("chunks", cfg.chunks_collection), ("item_names", cfg.item_name_collection)):
                if coll and mc.has_collection(coll):
                    res = mc.delete(collection_name=coll, filter=expr)
                    purged[key] = res.get("delete_count", 0) if isinstance(res, dict) else 0
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"清理向量库失败: {e}")

        kb_registry.delete_kb(kb_id)
        return {"kb_id": kb_id, "deleted": True, "purged": purged}


if __name__ == '__main__':
    # param1:fastapi实例
    # param2:启动的服务器地址
    # param3:启动的服务端口
    uvicorn.run(app=create_app(), host="0.0.0.0", port=int(os.getenv("IMPORT_PORT", "8002")), log_level="info")
