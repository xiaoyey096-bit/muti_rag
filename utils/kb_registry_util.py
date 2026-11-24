"""知识库注册表

多产品共用一套 Milvus 集合 / MinIO 桶 / Mongo 库，靠 kb_id 隔离。
本模块在 MongoDB 的 knowledge_bases 集合里维护每个知识库（产品）的元数据：
产品名、前端展示名、示例提示语、已导入文档清单、商品名清单等。
"""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from pymongo import DESCENDING
from pymongo.collection import Collection

from muti_rag.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)

KB_COLLECTION = "knowledge_bases"


def _get_collection() -> Collection:
    return StorageClients.get_mongo_db()[KB_COLLECTION]


def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
    doc["_id"] = str(doc.get("_id", ""))
    return doc


def create_kb(product_name: str, display_name: str = "", hint: str = "",
              provisional: bool = False) -> str:
    """新建一个知识库，返回 kb_id。provisional=True 表示产品名暂用文件标题占位，
    等导入流程里 LLM 抽出真实商品名后再回填。"""
    kb_id = uuid.uuid4().hex[:12]
    now = datetime.now().timestamp()
    doc = {
        "kb_id": kb_id,
        "product_name": product_name,
        "display_name": display_name or product_name,
        "hint": hint or "",
        "doc_titles": [],
        "item_names": [],
        "suggested_questions": [],
        "provisional": provisional,
        "created_at": now,
        "updated_at": now,
    }
    _get_collection().insert_one(doc)
    logger.info(f"创建知识库 kb_id={kb_id} product_name={product_name} provisional={provisional}")
    return kb_id


def get_kb(kb_id: str) -> Optional[Dict[str, Any]]:
    if not kb_id:
        return None
    doc = _get_collection().find_one({"kb_id": kb_id})
    return _serialize(doc) if doc else None


def list_kbs() -> List[Dict[str, Any]]:
    cursor = _get_collection().find({}).sort("created_at", DESCENDING)
    return [_serialize(doc) for doc in cursor]


def ensure_kb(kb_id: Optional[str], product_name: Optional[str], file_title: str,
              display_name: Optional[str] = None, hint: Optional[str] = None) -> str:
    """上传时确定 kb_id：传入已有 kb_id 则追加文档；否则新建。
    未显式给产品名时用文件标题占位（provisional），待 LLM 抽取后回填。"""
    if kb_id and get_kb(kb_id):
        return kb_id
    provisional = not product_name
    name = product_name or file_title
    return create_kb(name, display_name or "", hint or "", provisional=provisional)


def register_doc(kb_id: str, file_title: str, item_name: str) -> None:
    """导入完成后把文档登记进知识库；若知识库仍是占位名，用 LLM 抽出的商品名回填。"""
    coll = _get_collection()
    kb = coll.find_one({"kb_id": kb_id})
    if not kb:
        logger.warning(f"登记文档失败：知识库不存在 kb_id={kb_id}")
        return
    sets: Dict[str, Any] = {"updated_at": datetime.now().timestamp()}
    if kb.get("provisional") and item_name:
        sets["product_name"] = item_name
        sets["display_name"] = item_name
        sets["provisional"] = False
    coll.update_one(
        {"kb_id": kb_id},
        {"$addToSet": {"doc_titles": file_title, "item_names": item_name}, "$set": sets},
    )
    logger.info(f"知识库 {kb_id} 登记文档《{file_title}》商品名={item_name}")


def update_kb_meta(kb_id: str, display_name: Optional[str] = None,
                   hint: Optional[str] = None, product_name: Optional[str] = None) -> None:
    sets: Dict[str, Any] = {"updated_at": datetime.now().timestamp()}
    if display_name is not None:
        sets["display_name"] = display_name
    if hint is not None:
        sets["hint"] = hint
    if product_name is not None:
        sets["product_name"] = product_name
    _get_collection().update_one({"kb_id": kb_id}, {"$set": sets})


def set_suggested_questions(kb_id: str, questions: List[str]) -> None:
    """写入导入期 LLM 生成的"用户常问 3 问"，供问答入口页快速提问按钮。"""
    _get_collection().update_one(
        {"kb_id": kb_id},
        {"$set": {"suggested_questions": list(questions or []),
                  "updated_at": datetime.now().timestamp()}},
    )


def delete_kb(kb_id: str) -> int:
    """仅删除注册表记录（向量/对象数据清理由调用方按需另行处理）。"""
    result = _get_collection().delete_one({"kb_id": kb_id})
    return result.deleted_count