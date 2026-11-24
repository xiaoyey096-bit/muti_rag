"""
任务执行器（P2）：按步骤执行导入链，负责错误分类、重试调度、熔断协作与步骤审计。

与 P0 的 ResumeService 区别：
- 不再"整链 stream 跑完"，而是逐节点执行，每步成功才推进 last_success_step；
- 失败时区分 TRANSIENT / RESOURCE / PERMANENT，按步骤配额退避重试；
- 每步执行前检查依赖熔断器，熔断期间不消耗重试次数。
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from muti_rag.processor.import_processor.nodes.document_split_node import DocumentSplitNode
from muti_rag.processor.import_processor.nodes.embedding_chunks_node import EmbeddingChunksNode
from muti_rag.processor.import_processor.nodes.entry_node import EntryNode
from muti_rag.processor.import_processor.nodes.import_milvus_node import ImportMilvusNode
from muti_rag.processor.import_processor.nodes.item_name_recognition_node import ItemNameRecognitionNode
from muti_rag.processor.import_processor.nodes.md_to_img_node import MarkDownToImgNode
from muti_rag.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode
from muti_rag.processor.import_processor.nodes.suggest_questions_node import SuggestQuestionsNode
from muti_rag.utils import task_util
from muti_rag.utils.resilience import (
    PERMANENT, RESILIENCE, RESOURCE, STEP_MAX_ATTEMPTS, RESOURCE_EXTRA_ATTEMPTS,
    TASK_MAX_ATTEMPTS, TASK_MAX_NO_PROGRESS, backoff_delay, classify_error,
    step_semaphore, slog, user_message,
)

logger = logging.getLogger(__name__)

STEP_ORDER = [
    "entry_node", "pdf_to_md_node", "md_to_img_node", "document_split_node",
    "item_name_recognition_node", "embedding_chunk_node", "import_milvus_node",
    "suggest_questions_node",
]

STEP_TITLES = {
    "entry_node": "检查文件",
    "pdf_to_md_node": "PDF 转 Markdown",
    "md_to_img_node": "图片处理",
    "document_split_node": "文档切分",
    "item_name_recognition_node": "主体名称识别",
    "embedding_chunk_node": "向量生成",
    "import_milvus_node": "写入向量库",
    "suggest_questions_node": "推荐问题生成",
}


def _new_node(step: str):
    # 每次执行新建节点实例：节点本身无状态，但避免长驻线程间共享实例带来的隐式耦合
    return {
        "entry_node": EntryNode,
        "pdf_to_md_node": PdfToMdNode,
        "md_to_img_node": MarkDownToImgNode,
        "document_split_node": DocumentSplitNode,
        "item_name_recognition_node": ItemNameRecognitionNode,
        "embedding_chunk_node": EmbeddingChunksNode,
        "import_milvus_node": ImportMilvusNode,
        "suggest_questions_node": SuggestQuestionsNode,
    }[step]()


class TaskExecutor:

    def __init__(self, worker_id: str = "worker-0"):
        self.worker_id = worker_id

    # ==================== 主流程 ====================

    def execute(self, task_id: str) -> None:
        doc = task_util.get_task_doc(task_id)
        if not doc:
            return
        try:
            self._run(task_id, doc)
        except Exception as e:
            logger.exception(f"[{task_id}] 执行器异常退出")
            task_util.mark_failed_final(task_id, f"{user_message('PERMANENT')}（{e}）")
        finally:
            task_util.release_lease(task_id)

    def _run(self, task_id: str, doc: Dict[str, Any]) -> None:
        file_path = doc.get("import_file_path") or ""
        if not file_path or not os.path.exists(file_path):
            task_util.mark_failed_final(task_id, "原始文件已不存在，请重新上传")
            return

        state = self._build_state(doc)
        is_pdf = Path(file_path).suffix.lower() == ".pdf"
        steps = [s for s in STEP_ORDER if s != "pdf_to_md_node" or is_pdf]
        done = set(doc.get("done_nodes") or [])
        trace_id = doc.get("task_id", task_id)

        slog("task_start", task_id=task_id, trace_id=trace_id, worker_id=self.worker_id,
             steps=steps, resume_count=doc.get("resume_count", 0))

        for step in steps:
            if self._artifact_ready(step, state) and step in done:
                self._restore_state(step, state, doc)
                continue

            ok = self._run_step(task_id, step, state, doc, trace_id)
            if not ok[0]:
                return  # 已安排重试或终态失败，交给队列调度
            state = ok[1]

        task_util.mark_completed(task_id)
        slog("task_completed", task_id=task_id, trace_id=trace_id, chunks=len(state.get("chunks") or []))

    # ==================== 单步执行 ====================

    def _run_step(self, task_id: str, step: str, state: Dict[str, Any],
                  doc: Dict[str, Any], trace_id: str):
        """返回 (是否继续, 最新 state)。失败时已完成重试调度或终态标记。"""
        allowed, blocked_dep = RESILIENCE.allow_step(step)
        if not allowed:
            delay = 30.0
            logger.warning(f"[{task_id}] 熔断中（{blocked_dep}），{STEP_TITLES[step]} 延后 {delay:.0f}s 重排")
            task_util.schedule_retry(
                task_id, step,
                error=f"依赖服务 {blocked_dep} 暂时不可用，正在自动恢复",
                error_class="TRANSIENT", delay=delay, consume_attempt=False,
            )
            return False, state

        attempts_used = int((doc.get("step_attempts") or {}).get(step, 0)) + 1
        task_util.mark_step_started(task_id, step)
        slog("step_start", task_id=task_id, trace_id=trace_id, step=step,
             attempt=attempts_used, worker_id=self.worker_id)
        started = __import__("time").time()

        try:
            sem = step_semaphore(step)
            if sem is not None:
                with sem:
                    result = _new_node(step)(state)
            else:
                result = _new_node(step)(state)
            if isinstance(result, dict):
                state = result

            finished = __import__("time").time()
            task_util.record_step_attempt(task_id, step, attempts_used, "succeeded",
                                          started, finished, worker_id=self.worker_id)
            task_util.mark_step_succeeded(task_id, step)
            task_util.save_task_artifacts(task_id, state)
            RESILIENCE.record_step(step, True)
            slog("step_success", task_id=task_id, trace_id=trace_id, step=step,
                 attempt=attempts_used, duration_ms=int((finished - started) * 1000))
            return True, state

        except Exception as e:
            finished = __import__("time").time()
            error_class = classify_error(e)
            message = user_message(error_class) + f"（{e}）"
            task_util.record_step_attempt(task_id, step, attempts_used, "failed",
                                          started, finished, error_class=error_class,
                                          error_message=str(e), worker_id=self.worker_id)
            if error_class != PERMANENT:
                RESILIENCE.record_step(step, False, str(e))
            slog("step_failed", level=logging.ERROR, task_id=task_id, trace_id=trace_id,
                 step=step, attempt=attempts_used, error_class=error_class, error=str(e),
                 duration_ms=int((finished - started) * 1000))

            # suggest 失败不阻断整体导入（现状策略保留）
            if step == "suggest_questions_node":
                logger.warning(f"[{task_id}] 推荐问题生成失败，跳过：{e}")
                return True, state

            base = STEP_MAX_ATTEMPTS.get(step, 3)
            max_attempts = base + (RESOURCE_EXTRA_ATTEMPTS if error_class == RESOURCE else 0)
            task_attempts = int(doc.get("task_attempts") or 0) + 1
            no_progress = (int(doc.get("no_progress") or 0) + 1
                           if doc.get("last_failure_step") == step else 1)

            if error_class == PERMANENT:
                task_util.mark_failed_final(task_id, message, PERMANENT)
                return False, state
            if attempts_used >= max_attempts or task_attempts >= TASK_MAX_ATTEMPTS:
                task_util.mark_failed_final(
                    task_id,
                    f"{STEP_TITLES[step]} 已重试 {attempts_used} 次仍未成功。{message}", error_class)
                return False, state
            if no_progress >= TASK_MAX_NO_PROGRESS:
                task_util.mark_failed_final(
                    task_id,
                    f"{STEP_TITLES[step]} 连续 {no_progress} 次失败且无进展，已停止自动重试。{message}",
                    error_class)
                return False, state

            delay = backoff_delay(attempts_used)
            logger.warning(f"[{task_id}] {STEP_TITLES[step]} 第 {attempts_used} 次失败（{error_class}），"
                           f"{delay:.0f}s 后自动重试")
            task_util.schedule_retry(task_id, step, message, error_class, delay)
            return False, state

    # ==================== state 重建 / 跳过判定 ====================

    def _build_state(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        file_path = doc.get("import_file_path") or ""
        file_dir = doc.get("file_dir") or ""
        artifacts = doc.get("artifacts") or {}
        suffix = Path(file_path).suffix.lower()

        state: Dict[str, Any] = {
            "task_id": doc.get("task_id", ""),
            "import_file_path": file_path,
            "file_dir": file_dir,
            "kb_id": doc.get("kb_id", ""),
            "file_sha256": doc.get("file_sha256", ""),
            "file_title": doc.get("file_title") or Path(file_path).stem,
        }
        if suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = file_path
            md_path = artifacts.get("md_path") or PdfToMdNode()._get_md_path(Path(file_path), Path(file_dir))
        else:
            state["is_md_read_enabled"] = True
            md_path = artifacts.get("md_path") or file_path
        state["md_path"] = md_path

        new_md = str(Path(md_path).with_name(Path(md_path).stem + "_new" + Path(md_path).suffix))
        for candidate in (new_md, md_path):
            if candidate and os.path.exists(candidate):
                try:
                    state["md_content"] = Path(candidate).read_text(encoding="utf-8")
                    break
                except Exception as e:
                    logger.warning(f"读取 md 失败 {candidate}: {e}")

        chunks_path = artifacts.get("chunks_path")
        if chunks_path and os.path.exists(chunks_path):
            try:
                with open(chunks_path, "r", encoding="utf-8") as f:
                    state["chunks"] = json.load(f).get("chunks") or []
            except Exception as e:
                logger.warning(f"读取断点 chunks 失败 {chunks_path}: {e}")

        if artifacts.get("item_name"):
            state["item_name"] = artifacts["item_name"]
        if artifacts.get("suggested_questions"):
            state["suggested_questions"] = artifacts["suggested_questions"]
        return state

    def _artifact_ready(self, step: str, state: Dict[str, Any]) -> bool:
        if step == "entry_node":
            return bool(state.get("import_file_path")) and os.path.exists(state["import_file_path"])
        if step == "pdf_to_md_node":
            md_path = state.get("md_path") or ""
            return bool(md_path) and os.path.exists(md_path) and os.path.getsize(md_path) > 0
        if step == "md_to_img_node":
            md_path = state.get("md_path") or ""
            if not md_path:
                return False
            new_md = Path(md_path).with_name(Path(md_path).stem + "_new" + Path(md_path).suffix)
            return new_md.exists() and new_md.stat().st_size > 0
        if step == "document_split_node":
            return bool(state.get("chunks"))
        if step == "item_name_recognition_node":
            return bool(state.get("item_name"))
        if step == "embedding_chunk_node":
            chunks = state.get("chunks") or []
            return bool(chunks) and bool(chunks[0].get("dense_vector") and chunks[0].get("sparse_vector"))
        return True  # import_milvus / suggest：无本地校验物，按 done 标记跳过

    @staticmethod
    def _restore_state(step: str, state: Dict[str, Any], doc: Dict[str, Any]) -> None:
        artifacts = doc.get("artifacts") or {}
        if step == "item_name_recognition_node" and not state.get("item_name"):
            state["item_name"] = artifacts.get("item_name") or doc.get("file_title", "")
        if step == "suggest_questions_node" and not state.get("suggested_questions"):
            state["suggested_questions"] = artifacts.get("suggested_questions") or []
