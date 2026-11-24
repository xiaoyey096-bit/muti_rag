import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from collections import defaultdict


"""
任务id: 主要追踪上传文件（任务）的状态流程的
上传一个文件就属于一个任务（唯一的任务id）

P0 起任务状态双写：进程内存（兼容旧逻辑）+ MongoDB import_tasks 集合（重启可恢复）。
Mongo 不可用时自动降级为纯内存，不影响导入主流程。
"""

logger = logging.getLogger(__name__)

# 进程内存兜底缓存（Mongo 不可用 / 非导入类任务使用）
_tasks_running_list: Dict[str, List[str]] = defaultdict(list)
_tasks_done_list: Dict[str, List[str]] = defaultdict(list)
_tasks_duration: Dict[str, Dict[str, float]] = defaultdict(dict)
_tasks_result: Dict[str, Dict[str, str]] = defaultdict(dict)
_tasks_status: Dict[str, str] = {}

TASK_STATUS_QUEUED = "queued"  # 已提交，等待后台执行
TASK_STATUS_PROCESSING = "processing"  # 任务处理中
TASK_STATUS_WAITING_RETRY = "waiting_retry"  # 步骤失败/熔断，等待退避后自动重试
TASK_STATUS_COMPLETED = "completed"  # 任务完成
TASK_STATUS_FAILED = "failed"  # 任务失败（可手动继续导入）

_TASK_COLLECTION = "import_tasks"
_STEPS_COLLECTION = "import_task_steps"
_ARTIFACTS_FILE = "import_chunks.json"

_index_ready = False
_steps_index_ready = False
_mongo_warned = False

_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "上传文件",
    "entry_node": "检查文件",
    "pdf_to_md_node": "PDF转Markdown",
    "md_to_img_node": "Markdown图片处理",
    "document_split_node": "文档切分",
    "item_name_recognition_node": "主体名称识别",
    "embedding_chunk_node": "向量生成",
    "import_milvus_node": "导入向量数据库",
    "suggest_questions_node": "推荐问题生成",
    "__end__": "处理完成",
    # --- Query 流程节点（kb/query_process/main_graph.py）---
    "item_name_confirmed_node": "查询理解与产品确认",
    "answer_output_node": "生成答案",
    "evidence_gate_node": "证据门控",
    "reranker_node": "重排序",
    "rrf_merge_node": "倒排融合",
    "web_mcp_search_node": "网络搜索",
    "local_search_node": "切片搜索(混合召回+按需假设文档)",
    "hybrid_vector_search_node": "切片搜索",
    "hyde_vector_search_node": "切片搜索(假设性文档)"
}


def _to_cn(node_name: str) -> str:
    # 1. 从节点映射字典中获取中文名，若未配置则直接返回原英文名
    return _NODE_NAME_TO_CN.get(node_name, node_name)


# ==================== Mongo 访问 ====================

def _get_collection():
    """
    获取 import_tasks 集合；Mongo 不可用时返回 None（调用方降级为内存）。
    """
    global _index_ready, _mongo_warned
    try:
        from muti_rag.utils.client.storage_clients import StorageClients
        coll = StorageClients.get_mongo_db()[_TASK_COLLECTION]
        if not _index_ready:
            coll.create_index("task_id", unique=True)
            coll.create_index([("created_at", 1)])
            coll.create_index([("status", 1), ("next_retry_at", 1)])
            coll.create_index([("status", 1), ("heartbeat_at", 1)])
            coll.create_index([("file_sha256", 1), ("kb_id", 1), ("file_title", 1)])
            _index_ready = True
        return coll
    except Exception as e:
        if not _mongo_warned:
            logger.warning(f"任务持久化存储不可用，降级为进程内存: {e}")
            _mongo_warned = True
        return None


def _get_steps_collection():
    """import_task_steps：每次步骤尝试一条审计记录。"""
    global _steps_index_ready
    try:
        from muti_rag.utils.client.storage_clients import StorageClients
        coll = StorageClients.get_mongo_db()[_STEPS_COLLECTION]
        if not _steps_index_ready:
            coll.create_index([("task_id", 1), ("step", 1), ("attempt", 1)], unique=True)
            coll.create_index([("step", 1), ("status", 1), ("finished_at", -1)])
            _steps_index_ready = True
        return coll
    except Exception:
        return None


def _db_write(task_id: str, update: Dict[str, Any], upsert: bool = False) -> None:
    coll = _get_collection()
    if coll is None:
        return
    try:
        coll.update_one({"task_id": task_id}, update, upsert=upsert)
    except Exception as e:
        logger.warning(f"任务状态写入失败 task={task_id}: {e}")


_INSERT_DEFAULTS = {
    "done_nodes": [],
    "running_nodes": [],
    "durations": {},
    "artifacts": {},
    "error": "",
    "resumable": False,
    "resume_count": 0,
    # --- P2：租约 / 重试 / 步骤计数 ---
    "file_id": "",
    "step_attempts": {},
    "task_attempts": 0,
    "no_progress": 0,
    "last_failure_step": "",
    "last_success_step": "",
    "current_step": "",
    "crash_count": 0,
    "worker_id": "",
    "lease_until": 0.0,
    "heartbeat_at": 0.0,
    "next_retry_at": 0.0,
    "error_class": "",
    "trace_id": "",
    "status_updated_at": 0.0,
}


def _on_insert_defaults(exclude: tuple = (), status: str = TASK_STATUS_QUEUED) -> Dict[str, Any]:
    """upsert 时初始化缺失字段；exclude 用于避开与 $set/$addToSet/$pull 冲突的路径。"""
    base = dict(_INSERT_DEFAULTS, status=status, created_at=time.time())
    for key in exclude:
        base.pop(key, None)
    return base


# ==================== 任务创建 / 恢复 ====================

def create_task(task_id: str, kb_id: str = "", file_title: str = "",
                import_file_path: str = "", file_dir: str = "",
                file_sha256: str = "", file_id: str = "") -> None:
    """上传完成后创建（或补全）任务文档，状态 queued。"""
    now = time.time()
    coll = _get_collection()
    if coll is None:
        _tasks_status[task_id] = TASK_STATUS_QUEUED
        return
    try:
        coll.update_one(
            {"task_id": task_id},
            {
                "$set": {
                    "kb_id": kb_id,
                    "file_title": file_title,
                    "import_file_path": import_file_path,
                    "file_dir": file_dir,
                    "file_sha256": file_sha256,
                    "file_id": file_id or file_sha256,
                    "status": TASK_STATUS_QUEUED,
                    "trace_id": task_id,
                    "status_updated_at": now,
                    "updated_at": now,
                },
                "$setOnInsert": _on_insert_defaults(
                    exclude=("status", "file_id", "trace_id", "status_updated_at")),
            },
            upsert=True,
        )
    except Exception as e:
        logger.warning(f"任务创建写入失败 task={task_id}: {e}")


def get_task_doc(task_id: str) -> Optional[Dict[str, Any]]:
    """读取任务文档（断点续跑 / 状态查询用）；不存在或不可用时返回 None。"""
    coll = _get_collection()
    if coll is None:
        return None
    try:
        return coll.find_one({"task_id": task_id}, {"_id": 0})
    except Exception as e:
        logger.warning(f"任务读取失败 task={task_id}: {e}")
        return None


def recover_stale_tasks(stale_seconds: int = 120, max_crashes: int = 3) -> int:
    """
    服务启动时调用：上次进程遗留的 processing 任务心跳已停，
    统一回 waiting_retry 由队列续跑；连续崩溃超过 max_crashes 次 → failed（防止死循环）。
    queued 任务不需要处理，队列启动扫描时会直接捡起。
    """
    coll = _get_collection()
    if coll is None:
        return 0
    cutoff = time.time() - stale_seconds
    recovered = 0
    try:
        stale = list(coll.find(
            {"status": TASK_STATUS_PROCESSING,
             "$or": [{"heartbeat_at": {"$lt": cutoff}}, {"heartbeat_at": {"$exists": False}}]},
            {"task_id": 1, "crash_count": 1},
        ))
        for doc in stale:
            crash_count = int(doc.get("crash_count") or 0) + 1
            if crash_count > max_crashes:
                coll.update_one({"task_id": doc["task_id"]}, {"$set": {
                    "status": TASK_STATUS_FAILED,
                    "error": f"任务连续 {crash_count} 次异常中断，已停止自动重试，请手动继续导入",
                    "error_class": "PERMANENT",
                    "resumable": True,
                    "running_nodes": [],
                    "worker_id": "",
                    "lease_until": 0,
                    "crash_count": crash_count,
                    "status_updated_at": time.time(),
                    "updated_at": time.time(),
                }})
            else:
                coll.update_one({"task_id": doc["task_id"]}, {"$set": {
                    "status": TASK_STATUS_WAITING_RETRY,
                    "error": "服务重启导致导入中断，正在自动恢复",
                    "error_class": "TRANSIENT",
                    "next_retry_at": time.time(),
                    "resumable": False,
                    "running_nodes": [],
                    "worker_id": "",
                    "lease_until": 0,
                    "crash_count": crash_count,
                    "status_updated_at": time.time(),
                    "updated_at": time.time(),
                }})
            recovered += 1
        if recovered:
            logger.warning(f"启动恢复：{recovered} 个中断的导入任务已重新排队")
        return recovered
    except Exception as e:
        logger.warning(f"启动恢复失败: {e}")
        return 0


def claim_task(task_id: str, worker_id: str, lease_seconds: int = 60) -> Optional[Dict[str, Any]]:
    """CAS 抢占：只有 queued / waiting_retry 的任务能被领取执行。"""
    coll = _get_collection()
    if coll is None:
        return None
    now = time.time()
    try:
        return coll.find_one_and_update(
            {"task_id": task_id, "status": {"$in": [TASK_STATUS_QUEUED, TASK_STATUS_WAITING_RETRY]}},
            {"$set": {
                "status": TASK_STATUS_PROCESSING,
                "worker_id": worker_id,
                "lease_until": now + lease_seconds,
                "heartbeat_at": now,
                "next_retry_at": 0.0,
                "error": "",
                "error_class": "",
                "resumable": False,
                "running_nodes": [],
                "status_updated_at": now,
                "updated_at": now,
            }},
            return_document=True,
            projection={"_id": 0},
        )
    except Exception as e:
        logger.warning(f"任务抢占失败 task={task_id}: {e}")
        return None


def heartbeat(task_id: str, worker_id: str, lease_seconds: int = 60) -> None:
    coll = _get_collection()
    if coll is None:
        return
    try:
        now = time.time()
        coll.update_one(
            {"task_id": task_id, "worker_id": worker_id},
            {"$set": {"heartbeat_at": now, "lease_until": now + lease_seconds}},
        )
    except Exception:
        pass


def release_lease(task_id: str) -> None:
    _db_write(task_id, {"$set": {"worker_id": "", "lease_until": 0}})


def mark_step_started(task_id: str, step: str) -> None:
    _db_write(task_id, {"$set": {"current_step": step, "updated_at": time.time()}})


def mark_step_succeeded(task_id: str, step: str) -> None:
    """步骤成功：记录 last_success_step，并清零无进展计数。"""
    _db_write(task_id, {"$set": {
        "last_success_step": step,
        "no_progress": 0,
        "updated_at": time.time(),
    }})


def schedule_retry(task_id: str, step: str, error: str, error_class: str, delay: float,
                   consume_attempt: bool = True) -> None:
    """步骤失败可重试：回 waiting_retry，退避后由调度线程重新出队。
    consume_attempt=False 用于熔断等待：占位重排但不消耗该步骤的重试配额。"""
    now = time.time()
    _tasks_status[task_id] = TASK_STATUS_WAITING_RETRY
    coll = _get_collection()
    if coll is None:
        return
    try:
        update: Dict[str, Any] = {
            "$set": {
                "status": TASK_STATUS_WAITING_RETRY,
                "error": error,
                "error_class": error_class,
                "next_retry_at": now + delay,
                "resumable": False,
                "running_nodes": [],
                "current_step": step,
                "worker_id": "",
                "lease_until": 0,
                "status_updated_at": now,
                "updated_at": now,
            },
        }
        if consume_attempt:
            doc = coll.find_one({"task_id": task_id}, {"last_failure_step": 1, "no_progress": 1})
            same_step = bool(doc) and doc.get("last_failure_step") == step
            no_progress = int((doc or {}).get("no_progress") or 0) + 1 if same_step else 1
            update["$set"]["last_failure_step"] = step
            update["$set"]["no_progress"] = no_progress
            update["$inc"] = {"step_attempts." + step: 1, "task_attempts": 1}
        coll.update_one({"task_id": task_id}, update)
    except Exception as e:
        logger.warning(f"重试调度写入失败 task={task_id}: {e}")


def mark_failed_final(task_id: str, error: str, error_class: str = "PERMANENT") -> None:
    """不可重试（或重试耗尽）：failed + resumable，等待用户手动"继续导入"。"""
    now = time.time()
    _tasks_status[task_id] = TASK_STATUS_FAILED
    _db_write(task_id, {
        "$set": {
            "status": TASK_STATUS_FAILED,
            "error": error,
            "error_class": error_class,
            "resumable": True,
            "running_nodes": [],
            "current_step": "",
            "worker_id": "",
            "lease_until": 0,
            "status_updated_at": now,
            "updated_at": now,
        },
        "$setOnInsert": _on_insert_defaults(exclude=(
            "status", "error", "resumable", "running_nodes", "current_step",
            "worker_id", "lease_until", "status_updated_at")),
    }, upsert=True)


def mark_completed(task_id: str) -> None:
    now = time.time()
    _tasks_status[task_id] = TASK_STATUS_COMPLETED
    _db_write(task_id, {"$set": {
        "status": TASK_STATUS_COMPLETED,
        "error": "",
        "error_class": "",
        "resumable": False,
        "running_nodes": [],
        "current_step": "",
        "worker_id": "",
        "lease_until": 0,
        "finished_at": now,
        "status_updated_at": now,
        "updated_at": now,
    }})


def inc_crash_count(task_id: str) -> int:
    coll = _get_collection()
    if coll is None:
        return 0
    try:
        doc = coll.find_one_and_update({"task_id": task_id}, {"$inc": {"crash_count": 1}},
                                       return_document=True, projection={"crash_count": 1})
        return int((doc or {}).get("crash_count") or 0)
    except Exception:
        return 0


def request_resume(task_id: str) -> None:
    """用户点"继续导入"：failed → queued，清空重试计数，交给队列。"""
    _tasks_status[task_id] = TASK_STATUS_QUEUED
    _db_write(task_id, {
        "$set": {
            "status": TASK_STATUS_QUEUED,
            "error": "",
            "error_class": "",
            "resumable": False,
            "running_nodes": [],
            "current_step": "",
            "step_attempts": {},
            "task_attempts": 0,
            "no_progress": 0,
            "crash_count": 0,
            "next_retry_at": 0.0,
            "worker_id": "",
            "lease_until": 0,
            "status_updated_at": time.time(),
            "updated_at": time.time(),
        },
        "$inc": {"resume_count": 1},
    })


def record_step_attempt(task_id: str, step: str, attempt: int, status: str,
                        started_at: float, finished_at: float,
                        error_class: str = "", error_message: str = "",
                        worker_id: str = "") -> None:
    """写一条 import_task_steps 审计记录（一次尝试一条）。"""
    coll = _get_steps_collection()
    if coll is None:
        return
    try:
        coll.update_one(
            {"task_id": task_id, "step": step, "attempt": attempt},
            {"$set": {
                "task_id": task_id, "step": step, "attempt": attempt, "status": status,
                "started_at": started_at, "finished_at": finished_at,
                "duration_ms": int(max(0.0, finished_at - started_at) * 1000),
                "error_class": error_class, "error_message": error_message[:1000],
                "worker_id": worker_id,
            }},
            upsert=True,
        )
    except Exception as e:
        logger.warning(f"步骤记录写入失败 task={task_id} step={step}: {e}")


def list_tasks_by_status(statuses: List[str], limit: int = 50) -> List[Dict[str, Any]]:
    coll = _get_collection()
    if coll is None:
        return []
    try:
        return list(coll.find({"status": {"$in": statuses}}, {"_id": 0})
                    .sort("created_at", 1).limit(limit))
    except Exception:
        return []


def list_due_retry_tasks(now: float = None, limit: int = 20) -> List[Dict[str, Any]]:
    coll = _get_collection()
    if coll is None:
        return []
    now = now or time.time()
    try:
        return list(coll.find(
            {"status": TASK_STATUS_WAITING_RETRY, "next_retry_at": {"$lte": now}},
            {"task_id": 1, "_id": 0},
        ).sort("next_retry_at", 1).limit(limit))
    except Exception:
        return []


# ==================== 节点进度（双写） ====================

def add_running_task(task_id: str, node_name: str) -> None:
    # 1. 获取当前任务的运行节点列表（利用 defaultdict 自动初始化特性）
    running = _tasks_running_list[task_id]

    # 2. 将当前节点加入运行列表（并做去重判断，防止重复添加）
    if node_name not in running:
        running.append(node_name)

    # 3. 持久化（setOnInsert 避开 running_nodes，避免与 $addToSet 路径冲突）
    _db_write(task_id, {
        "$addToSet": {"running_nodes": node_name},
        "$set": {"updated_at": time.time()},
        "$setOnInsert": _on_insert_defaults(exclude=("running_nodes",)),
    }, upsert=True)


def add_done_task(task_id: str, node_name: str) -> None:
    # 1. 如果该节点还在运行列表中，则将其移出（表示该节点已结束运行）
    if node_name in _tasks_running_list[task_id]:
        _tasks_running_list[task_id].remove(node_name)

    # 2. 获取当前任务的已完成节点列表
    done = _tasks_done_list[task_id]

    # 3. 将当前节点加入已完成列表（做去重判断，防止重复标记）
    if node_name not in done:
        done.append(node_name)

    # 4. 持久化（setOnInsert 避开 running_nodes/done_nodes，避免路径冲突）
    _db_write(task_id, {
        "$pull": {"running_nodes": node_name},
        "$addToSet": {"done_nodes": node_name},
        "$set": {"updated_at": time.time()},
        "$setOnInsert": _on_insert_defaults(exclude=("running_nodes", "done_nodes")),
    }, upsert=True)


def add_node_duration(task_id: str, node_name: str, duration: float) -> None:
    """记录节点耗时（秒），展示名按中文存储。"""
    cn_name = _to_cn(node_name)
    _tasks_duration[task_id][cn_name] = round(duration, 2)
    _db_write(task_id, {"$set": {f"durations.{cn_name}": round(duration, 2)}})


def save_task_artifacts(task_id: str, state: Dict[str, Any]) -> None:
    """
    把断点续跑所需的关键产物落盘/落库：
    - chunks 写 <file_dir>/import_chunks.json（含向量，避免续跑重算嵌入）
    - md_path / item_name / file_title / suggested_questions 写任务文档 artifacts
    """
    if not isinstance(state, dict):
        return
    art: Dict[str, Any] = {}
    try:
        file_dir = state.get("file_dir")
        chunks = state.get("chunks")
        if file_dir and isinstance(chunks, list) and chunks:
            path = os.path.join(file_dir, _ARTIFACTS_FILE)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"chunks": chunks}, f, ensure_ascii=False)
            os.replace(tmp, path)
            art["chunks_path"] = path
            art["chunks_count"] = len(chunks)
        for key in ("md_path", "item_name", "file_title"):
            value = state.get(key)
            if value:
                art[key] = value
        if state.get("suggested_questions"):
            art["suggested_questions"] = list(state["suggested_questions"])
    except Exception as e:
        logger.warning(f"任务产物保存失败 task={task_id}: {e}")
        return
    if art:
        _db_write(task_id, {"$set": {f"artifacts.{k}": v for k, v in art.items()}})


# ==================== 状态读写 ====================

def get_running_task_list(task_id: str) -> List[str]:
    # 1. 进程内存优先
    running = _tasks_running_list.get(task_id, [])
    if not running:
        doc = get_task_doc(task_id)
        if doc:
            running = doc.get("running_nodes") or []
    return [_to_cn(n) for n in running]


def get_done_task_list(task_id: str) -> List[str]:
    done = _tasks_done_list.get(task_id, [])
    if not done:
        doc = get_task_doc(task_id)
        if doc:
            done = doc.get("done_nodes") or []
    return [_to_cn(n) for n in done]


def get_task_status(task_id: str) -> str:
    """
    根据任务ID 获取任务状态
    :param task_id:
    :return:
    """
    if task_id in _tasks_status:
        return _tasks_status[task_id]
    doc = get_task_doc(task_id)
    return (doc or {}).get("status", "")


def update_task_status(task_id: str, status_name: str, error: Optional[str] = None) -> None:
    """更新任务总体状态；失败时附带错误信息并标记可续跑。"""
    _tasks_status[task_id] = status_name
    now = time.time()
    fields: Dict[str, Any] = {"status": status_name, "updated_at": now, "status_updated_at": now}
    if error is not None:
        fields["error"] = error
    if status_name == TASK_STATUS_FAILED:
        fields["resumable"] = True
        fields["running_nodes"] = []
        fields["worker_id"] = ""
        fields["lease_until"] = 0
    elif status_name == TASK_STATUS_COMPLETED:
        fields["resumable"] = False
        if error is None:
            fields["error"] = ""
    else:
        fields["resumable"] = False
    _db_write(task_id, {
        "$set": fields,
        "$setOnInsert": _on_insert_defaults(exclude=(
            "status", "error", "resumable", "running_nodes", "current_step",
            "worker_id", "lease_until", "status_updated_at")),
    }, upsert=True)


def set_task_result(task_id: str, key: str, value: str) -> None:
    """
    存储任务结果字段（如 answer / error）。
    """
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """
    获取任务结果字段（如 answer / error）。
    """
    return _tasks_result.get(task_id, {}).get(key, default)


def get_node_durations(task_id: str) -> Dict[str, float]:
    """获取所有节点的耗时"""
    durations = _tasks_duration.get(task_id, {})
    if not durations:
        doc = get_task_doc(task_id)
        if doc:
            durations = doc.get("durations") or {}
    return dict(durations)


def get_task_info(task_id: str) -> Dict[str, Any]:
    """
    获取任务的全局信息（状态 + 运行中节点 + 已完成节点）
    :param task_id: 任务ID
    :return: 包含 status、running_list、done_list 的字典
    """
    doc = get_task_doc(task_id)
    if doc:
        status = doc.get("status", "")
        next_retry_at = float(doc.get("next_retry_at") or 0)
        return {
            "status": status,
            "running_list": [_to_cn(n) for n in (doc.get("running_nodes") or [])],
            "done_list": [_to_cn(n) for n in (doc.get("done_nodes") or [])],
            "durations": dict(doc.get("durations") or {}),
            "error": doc.get("error", ""),
            "resumable": bool(doc.get("resumable")),
            "resume_count": int(doc.get("resume_count") or 0),
            "current_step": _to_cn(doc.get("current_step") or ""),
            "error_class": doc.get("error_class", ""),
            "attempts": {_to_cn(k): int(v) for k, v in (doc.get("step_attempts") or {}).items()},
            "next_retry_in": max(0, int(next_retry_at - time.time())) if status == TASK_STATUS_WAITING_RETRY else 0,
        }
    # 无任务文档（旧任务或非导入类任务）→ 退回进程内存
    return {
        "status": _tasks_status.get(task_id, ""),
        "running_list": [_to_cn(n) for n in _tasks_running_list.get(task_id, [])],
        "done_list": [_to_cn(n) for n in _tasks_done_list.get(task_id, [])],
        "durations": dict(_tasks_duration.get(task_id, {})),
        "error": "",
        "resumable": False,
        "resume_count": 0,
        "current_step": "",
        "error_class": "",
        "attempts": {},
        "next_retry_in": 0,
    }
