"""
临时数据清理（P2）：
- 过期/中止/完成的上传会话及其分片目录；
- 没有会话引用的孤儿分片目录；
- 终态任务（completed/failed）超过保留期的任务目录。

由任务队列的 janitor 线程周期调用，也可手动 `python -m muti_rag.utils.cleanup_util`。
"""

import logging
import os
import shutil
import time
from typing import Dict

from muti_rag.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)

TEMP_RETENTION_DAYS = int(os.getenv("TEMP_RETENTION_DAYS", "7"))


def _sessions_coll():
    return StorageClients.get_mongo_db()["upload_sessions"]


def _tasks_coll():
    return StorageClients.get_mongo_db()["import_tasks"]


def cleanup_uploads() -> Dict[str, int]:
    """清理过期会话与孤儿分片目录。"""
    from muti_rag.utils.upload_session_util import UPLOAD_ROOT, S_OPEN, S_ABORTED, S_COMPLETED

    stats = {"sessions": 0, "dirs": 0}
    now = time.time()
    try:
        coll = _sessions_coll()
        stale = list(coll.find({
            "$or": [
                {"status": S_OPEN, "expires_at": {"$lt": now}},
                {"status": {"$in": [S_ABORTED, S_COMPLETED]},
                 "updated_at": {"$lt": now - 24 * 3600}},
            ]
        }, {"upload_id": 1, "_id": 0}))
        for doc in stale:
            shutil.rmtree(os.path.join(UPLOAD_ROOT, doc["upload_id"]), ignore_errors=True)
            coll.delete_one({"upload_id": doc["upload_id"]})
            stats["sessions"] += 1

        if os.path.isdir(UPLOAD_ROOT):
            live = {d["upload_id"] for d in coll.find({}, {"upload_id": 1, "_id": 0})}
            for name in os.listdir(UPLOAD_ROOT):
                path = os.path.join(UPLOAD_ROOT, name)
                if name in live or not os.path.isdir(path):
                    continue
                if now - os.path.getmtime(path) > 24 * 3600:
                    shutil.rmtree(path, ignore_errors=True)
                    stats["dirs"] += 1
    except Exception as e:
        logger.warning(f"上传会话清理失败: {e}")
    return stats


def cleanup_task_dirs(days: int = TEMP_RETENTION_DAYS) -> int:
    """删除终态任务超过保留期的本地工作目录（产物已入库/MinIO）。"""
    removed = 0
    cutoff = time.time() - days * 86400
    try:
        coll = _tasks_coll()
        cursor = coll.find(
            {"status": {"$in": ["completed", "failed"]}, "updated_at": {"$lt": cutoff}},
            {"file_dir": 1, "_id": 0},
        )
        for doc in cursor:
            file_dir = doc.get("file_dir") or ""
            if file_dir and os.path.isdir(file_dir):
                shutil.rmtree(file_dir, ignore_errors=True)
                removed += 1
    except Exception as e:
        logger.warning(f"任务目录清理失败: {e}")
    return removed


def cleanup_all() -> Dict[str, int]:
    upload_stats = cleanup_uploads()
    task_dirs = cleanup_task_dirs()
    result = {"sessions": upload_stats["sessions"], "upload_dirs": upload_stats["dirs"],
              "task_dirs": task_dirs}
    if any(result.values()):
        logger.info(f"临时数据清理：{result}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(cleanup_all())
