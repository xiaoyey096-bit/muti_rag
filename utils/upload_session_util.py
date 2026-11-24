import hashlib
import logging
import math
import os
import shutil
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from muti_rag.core.paths import get_local_base_dir
from muti_rag.utils.client.storage_clients import StorageClients
from muti_rag.utils.kb_registry_util import ensure_kb
from muti_rag.utils.task_util import create_task, get_task_doc, TASK_STATUS_COMPLETED

logger = logging.getLogger(__name__)

SESSIONS_COLLECTION = "upload_sessions"
FILES_COLLECTION = "files"

UPLOAD_ROOT = os.path.join(get_local_base_dir(), "_uploads")
FILES_ROOT = os.path.join(get_local_base_dir(), "files")

CHUNK_SIZE = int(float(os.getenv("UPLOAD_CHUNK_SIZE_MB", "5")) * 1024 * 1024)
SESSION_TTL_HOURS = int(os.getenv("UPLOAD_SESSION_TTL_HOURS", "24"))
MIN_FREE_MB = int(os.getenv("UPLOAD_MIN_FREE_MB", "2048"))

S_OPEN = "OPEN"
S_MERGING = "MERGING"
S_COMPLETED = "COMPLETED"
S_ABORTED = "ABORTED"

_index_ready = False


def _collections():
    global _index_ready
    db = StorageClients.get_mongo_db()
    sessions = db[SESSIONS_COLLECTION]
    files = db[FILES_COLLECTION]
    if not _index_ready:
        sessions.create_index("upload_id", unique=True)
        sessions.create_index("sha256")
        sessions.create_index("expires_at", expireAfterSeconds=0)
        files.create_index("file_id", unique=True)
        files.create_index("sha256", unique=True)
        _index_ready = True
    return sessions, files


# ==================== 基础工具 ====================

def safe_filename(filename: str) -> str:
    """只取基名，禁止路径穿越。"""
    name = os.path.basename((filename or "").replace("\\", "/")).strip()
    return name or f"upload_{uuid.uuid4().hex[:8]}"


def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(1024 * 1024)
            if not buf:
                break
            hasher.update(buf)
    return hasher.hexdigest()


def free_bytes(path: str = None) -> int:
    try:
        return shutil.disk_usage(path or get_local_base_dir()).free
    except Exception:
        return -1


def _part_path(upload_id: str, part_no: int) -> str:
    return os.path.join(UPLOAD_ROOT, upload_id, f"part_{part_no:05d}")


def _session_public(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "upload_id": doc.get("upload_id", ""),
        "filename": doc.get("filename", ""),
        "size": doc.get("size", 0),
        "sha256": doc.get("sha256", ""),
        "chunk_size": doc.get("chunk_size", CHUNK_SIZE),
        "total_parts": doc.get("total_parts", 0),
        "received_parts": sorted(doc.get("received_parts") or []),
        "status": doc.get("status", ""),
        "file_id": doc.get("file_id", ""),
        "created_at": doc.get("created_at", 0),
        "expires_at": doc.get("expires_at", 0),
    }


# ==================== 文件层 ====================

def init_upload(filename: str, size: int, sha256: str = "",
                chunk_size: int = 0, kb_id: str = "", product_name: str = "") -> Dict[str, Any]:
    """
    - 同 sha256 已有完整文件 → 秒传（status=COMPLETED + file_id）
    - 同 sha256 有 OPEN 会话 → 返回已有会话（断点信息）
    - 否则新建会话
    """
    sessions, files = _collections()
    filename = safe_filename(filename)
    sha256 = (sha256 or "").strip().lower()
    chunk_size = int(chunk_size or CHUNK_SIZE)
    chunk_size = max(1, min(chunk_size, 32 * 1024 * 1024))
    now = time.time()

    if sha256:
        existing = files.find_one({"sha256": sha256}, {"_id": 0})
        if existing:
            files.update_one({"file_id": existing["file_id"]}, {"$inc": {"ref_count": 1}})
            return {
                "completed": True, "upload_id": "", "file_id": existing["file_id"],
                "filename": existing.get("filename", filename), "size": existing.get("size", size),
                "chunk_size": chunk_size, "total_parts": 0, "received_parts": [],
                "status": S_COMPLETED, "expires_at": 0,
            }
        existing = sessions.find_one({"sha256": sha256, "status": S_OPEN, "size": int(size)}, {"_id": 0})
        if existing:
            sessions.update_one({"upload_id": existing["upload_id"]},
                                {"$inc": {"resume_hits": 1}, "$set": {"updated_at": now}})
            logger.info(f"断点续传命中：session={existing['upload_id']} "
                        f"已收 {len(existing.get('received_parts') or [])}/{existing.get('total_parts')}")
            return dict(_session_public(existing), completed=False, resumed=True)

    total_parts = max(1, math.ceil(int(size) / chunk_size))
    doc = {
        "upload_id": uuid.uuid4().hex,
        "filename": filename,
        "size": int(size),
        "sha256": sha256,
        "chunk_size": chunk_size,
        "total_parts": total_parts,
        "received_parts": [],
        "status": S_OPEN,
        "file_id": "",
        "kb_id": kb_id,
        "product_name": product_name,
        "resume_hits": 0,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + SESSION_TTL_HOURS * 3600,
    }
    sessions.insert_one(doc)
    os.makedirs(os.path.join(UPLOAD_ROOT, doc["upload_id"]), exist_ok=True)
    return dict(_session_public(doc), completed=False)


def get_upload(upload_id: str) -> Optional[Dict[str, Any]]:
    sessions, _ = _collections()
    doc = sessions.find_one({"upload_id": upload_id}, {"_id": 0})
    return doc


def save_part(upload_id: str, part_no: int, data: bytes) -> Dict[str, Any]:
    sessions, _ = _collections()
    session = sessions.find_one({"upload_id": upload_id}, {"_id": 0})
    if not session:
        raise KeyError("上传会话不存在或已过期")
    if session.get("status") != S_OPEN:
        raise RuntimeError(f"上传会话状态为 {session.get('status')}，不可写入分片")
    total = int(session.get("total_parts") or 0)
    if not (1 <= part_no <= total):
        raise ValueError(f"分片编号越界：{part_no}/{total}")
    chunk_size = int(session.get("chunk_size") or CHUNK_SIZE)
    if len(data) > chunk_size:
        raise ValueError(f"分片过大：{len(data)} > {chunk_size}")

    path = _part_path(upload_id, part_no)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

    sessions.update_one({"upload_id": upload_id},
                        {"$addToSet": {"received_parts": part_no},
                         "$set": {"updated_at": time.time()}})
    count = sessions.find_one({"upload_id": upload_id}, {"received_parts": 1}).get("received_parts", [])
    return {"upload_id": upload_id, "part_no": part_no, "received_parts": sorted(count),
            "count": len(count), "total_parts": total}


def _merge_parts(upload_id: str, filename: str, total_parts: int) -> str:
    """顺序流式合并到持久 files 目录，返回本地路径。"""
    safe = safe_filename(filename)
    stem, ext = os.path.splitext(safe)
    tmp_path = os.path.join(UPLOAD_ROOT, upload_id, f"merged_{uuid.uuid4().hex}{ext}")
    hasher = hashlib.sha256()
    with open(tmp_path, "wb") as out:
        for part_no in range(1, total_parts + 1):
            with open(_part_path(upload_id, part_no), "rb") as part:
                while True:
                    buf = part.read(1024 * 1024)
                    if not buf:
                        break
                    hasher.update(buf)
                    out.write(buf)
    return tmp_path, hasher.hexdigest(), safe


def complete_upload(upload_id: str) -> Dict[str, Any]:
    """
    校验分片齐全 + 合并 + sha256 校验 + 登记 files + 上传 MinIO。
    返回 {"status": COMPLETED|OPEN|..., "file_id", "missing_parts", "error"}。
    """
    sessions, files = _collections()
    now = time.time()

    # CAS：OPEN → MERGING，防止并发双 complete
    session = sessions.find_one_and_update(
        {"upload_id": upload_id, "status": S_OPEN},
        {"$set": {"status": S_MERGING, "updated_at": now}},
        projection={"_id": 0},
    )
    if not session:
        session = sessions.find_one({"upload_id": upload_id}, {"_id": 0})
        if not session:
            raise KeyError("上传会话不存在或已过期")
        return {"status": session.get("status"), "file_id": session.get("file_id", ""),
                "missing_parts": [], "error": ""}

    total = int(session.get("total_parts") or 0)
    received = set(session.get("received_parts") or [])
    missing = [p for p in range(1, total + 1) if p not in received]
    if missing:
        sessions.update_one({"upload_id": upload_id}, {"$set": {"status": S_OPEN, "updated_at": now}})
        return {"status": S_OPEN, "file_id": "", "missing_parts": missing, "error": ""}

    try:
        merged_tmp, actual_sha, safe = _merge_parts(upload_id, session.get("filename", ""), total)
    except Exception as e:
        sessions.update_one({"upload_id": upload_id}, {"$set": {"status": S_OPEN, "updated_at": now}})
        raise RuntimeError(f"分片合并失败：{e}")

    expected = (session.get("sha256") or "").lower()
    if expected and actual_sha != expected:
        sessions.update_one({"upload_id": upload_id},
                            {"$set": {"status": S_OPEN, "updated_at": now,
                                      "last_sha_mismatch": {"expected": expected, "actual": actual_sha}}})
        return {"status": S_OPEN, "file_id": "", "missing_parts": [], "error": "sha256_mismatch",
                "expected": expected, "actual": actual_sha}

    # 持久文件路径：files/<sha2>/<sha>/<name>
    dest_dir = os.path.join(FILES_ROOT, actual_sha[:2], actual_sha)
    os.makedirs(dest_dir, exist_ok=True)
    local_path = os.path.join(dest_dir, safe)
    os.replace(merged_tmp, local_path)
    shutil.rmtree(os.path.join(UPLOAD_ROOT, upload_id), ignore_errors=True)

    minio_path = _upload_to_minio(local_path, safe, actual_sha)
    size = os.path.getsize(local_path)
    files.update_one(
        {"sha256": actual_sha},
        {"$setOnInsert": {"file_id": actual_sha, "sha256": actual_sha, "size": size,
                          "filename": safe, "local_path": local_path,
                          "minio_path": minio_path, "created_at": now},
         "$set": {"updated_at": now},
         "$inc": {"ref_count": 1}},
        upsert=True,
    )
    sessions.update_one({"upload_id": upload_id},
                        {"$set": {"status": S_COMPLETED, "file_id": actual_sha,
                                  "actual_sha256": actual_sha, "updated_at": now,
                                  "completed_at": now}})
    logger.info(f"分片上传完成：{safe} sha={actual_sha[:12]}… size={size} parts={total}")
    return {"status": S_COMPLETED, "file_id": actual_sha, "missing_parts": [], "error": "",
            "filename": safe, "size": size}


def abort_upload(upload_id: str) -> Dict[str, Any]:
    sessions, _ = _collections()
    now = time.time()
    sessions.update_one({"upload_id": upload_id},
                        {"$set": {"status": S_ABORTED, "updated_at": now}})
    shutil.rmtree(os.path.join(UPLOAD_ROOT, upload_id), ignore_errors=True)
    return {"upload_id": upload_id, "status": S_ABORTED}


def _upload_to_minio(local_path: str, filename: str, sha256: str) -> str:
    """按内容哈希归档原始文件（同日同名不再互相覆盖）。"""
    object_name = f"origin_files/{sha256[:2]}/{sha256}/{filename}"
    try:
        from muti_rag.utils.client.storage_clients import StorageClients as SC
        client = SC.get_minio_client()
        client.fput_object(os.getenv("MINIO_BUCKET_NAME"), object_name, local_path)
    except Exception as e:
        logger.warning(f"原始文件归档到 MinIO 失败（不阻断）：{e}")
    return object_name


def get_file(file_id: str) -> Optional[Dict[str, Any]]:
    _, files = _collections()
    return files.find_one({"file_id": file_id}, {"_id": 0})


def register_file(local_path: str, filename: str, sha256: str) -> Dict[str, Any]:
    """旧单请求上传路径：登记文件，供后续秒传命中。"""
    _, files = _collections()
    now = time.time()
    size = os.path.getsize(local_path)
    minio_path = _upload_to_minio(local_path, safe_filename(filename), sha256)
    files.update_one(
        {"sha256": sha256},
        {"$setOnInsert": {"file_id": sha256, "sha256": sha256, "size": size,
                          "filename": safe_filename(filename), "local_path": local_path,
                          "minio_path": minio_path, "created_at": now},
         "$set": {"updated_at": now},
         "$inc": {"ref_count": 1}},
        upsert=True,
    )
    return {"file_id": sha256}


# ==================== 业务层衔接 ====================

def create_task_for_file(file_id: str, kb_id: str = "", product_name: str = "") -> Dict[str, Any]:
    """
    以 file_id 创建导入任务（去重键：file_sha256 + kb_id + file_title）。
    返回 {"task_id", "kb_id", "deduplicated"}。
    """
    file_doc = get_file(file_id)
    if not file_doc:
        raise KeyError("文件不存在")

    filename = file_doc.get("filename") or safe_filename("upload")
    file_title = os.path.splitext(filename)[0]
    resolved_kb = ensure_kb(kb_id, product_name, file_title)

    db = StorageClients.get_mongo_db()
    existing = db["import_tasks"].find_one(
        {"file_sha256": file_doc["sha256"], "kb_id": resolved_kb, "file_title": file_title,
         "status": {"$in": ["queued", "processing", "waiting_retry", TASK_STATUS_COMPLETED]}},
        {"task_id": 1, "status": 1}, sort=[("created_at", -1)],
    )
    if existing:
        return {"task_id": existing["task_id"], "kb_id": resolved_kb,
                "status": existing.get("status", ""), "deduplicated": True}

    task_id = uuid.uuid4().hex[:8]
    file_dir = os.path.join(get_local_base_dir(), datetime.now().strftime("%Y%m%d"), task_id)
    os.makedirs(file_dir, exist_ok=True)
    import_file_path = os.path.join(file_dir, filename)

    # 同盘优先硬链接，避免复制 191MB；跨盘/失败回退复制
    try:
        os.link(file_doc["local_path"], import_file_path)
    except Exception:
        shutil.copy2(file_doc["local_path"], import_file_path)

    create_task(task_id, kb_id=resolved_kb, file_title=file_title,
                import_file_path=import_file_path, file_dir=file_dir,
                file_sha256=file_doc["sha256"])
    db["import_tasks"].update_one({"task_id": task_id}, {"$set": {"file_id": file_id}})

    from muti_rag.service.task_queue import get_task_queue
    get_task_queue().enqueue(task_id)

    return {"task_id": task_id, "kb_id": resolved_kb, "status": "queued", "deduplicated": False}
