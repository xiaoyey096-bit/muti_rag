"""
任务队列（P2）：进程内多线程 worker 从 Mongo 持久队列取任务执行。

- 队列数据在 Mongo（import_tasks），进程内 queue.Queue 只是唤醒通道；
- queued / waiting_retry 到期任务由调度线程重新入队；
- 启停恢复：启动时把上次遗留的 processing 任务回退为 waiting_retry（crash_count 超限则 failed）；
- 心跳线程每 10s 续租，janitor 线程每小时清理临时数据；
- 单进程默认 2 个 worker，GPU 解析在 step_semaphore 里全局串行。
"""

import logging
import os
import queue
import threading
import time
from typing import Dict, Optional, Set

from muti_rag.service.task_executor import TaskExecutor
from muti_rag.utils import task_util

logger = logging.getLogger(__name__)

WORKER_THREADS = int(os.getenv("TASK_WORKER_THREADS", "2"))
HEARTBEAT_INTERVAL = 10
SCHEDULER_INTERVAL = 2
JANITOR_INTERVAL = 3600


class TaskQueue:

    def __init__(self):
        self._q: "queue.Queue[str]" = queue.Queue()
        self._pending: Set[str] = set()
        self._active: Dict[str, str] = {}  # task_id -> worker_id
        self._lock = threading.Lock()
        self._started = False
        self._stopping = threading.Event()

    # ==================== 生命周期 ====================

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True

        recovered = task_util.recover_stale_tasks()
        queued = task_util.list_tasks_by_status([task_util.TASK_STATUS_QUEUED], limit=200)
        for doc in queued:
            self.enqueue(doc["task_id"])
        logger.info(f"任务队列启动：恢复 {recovered} 个中断任务，入队 {len(queued)} 个排队任务，"
                    f"worker={WORKER_THREADS}")

        for i in range(WORKER_THREADS):
            threading.Thread(target=self._worker_loop, args=(f"worker-{i + 1}",),
                             name=f"task-worker-{i + 1}", daemon=True).start()
        threading.Thread(target=self._scheduler_loop, name="task-scheduler", daemon=True).start()
        threading.Thread(target=self._heartbeat_loop, name="task-heartbeat", daemon=True).start()
        threading.Thread(target=self._janitor_loop, name="task-janitor", daemon=True).start()

    def stop(self) -> None:
        self._stopping.set()

    # ==================== 入队 ====================

    def enqueue(self, task_id: str) -> bool:
        with self._lock:
            if task_id in self._pending or task_id in self._active:
                return False
            self._pending.add(task_id)
        self._q.put(task_id)
        return True

    # ==================== 线程体 ====================

    def _worker_loop(self, worker_id: str) -> None:
        executor = TaskExecutor(worker_id)
        while not self._stopping.is_set():
            try:
                task_id = self._q.get(timeout=1)
            except queue.Empty:
                continue
            try:
                doc = task_util.claim_task(task_id, worker_id)
                if not doc:
                    continue
                with self._lock:
                    self._pending.discard(task_id)
                    self._active[task_id] = worker_id
                logger.info(f"[{task_id}] 由 {worker_id} 领取")
                executor.execute(task_id)
            except Exception:
                logger.exception(f"[{task_id}] worker 执行异常")
            finally:
                with self._lock:
                    self._active.pop(task_id, None)
                    self._pending.discard(task_id)
                self._q.task_done()

    def _scheduler_loop(self) -> None:
        while not self._stopping.wait(SCHEDULER_INTERVAL):
            try:
                for doc in task_util.list_due_retry_tasks(limit=20):
                    self.enqueue(doc["task_id"])
            except Exception as e:
                logger.warning(f"重试调度扫描失败: {e}")

    def _heartbeat_loop(self) -> None:
        while not self._stopping.wait(HEARTBEAT_INTERVAL):
            with self._lock:
                active = list(self._active.items())
            for task_id, worker_id in active:
                task_util.heartbeat(task_id, worker_id)

    def _janitor_loop(self) -> None:
        # 启动后先等 60s，避免和正在恢复的任务争磁盘 I/O
        if self._stopping.wait(60):
            return
        while True:
            try:
                from muti_rag.utils.cleanup_util import cleanup_all
                cleanup_all()
            except Exception as e:
                logger.warning(f"定时清理失败: {e}")
            if self._stopping.wait(JANITOR_INTERVAL):
                return

    # ==================== 观测 ====================

    def status(self) -> Dict[str, int]:
        with self._lock:
            return {"queue_depth": self._q.qsize(), "pending": len(self._pending),
                    "active": len(self._active), "workers": WORKER_THREADS}


_instance: Optional[TaskQueue] = None
_instance_lock = threading.Lock()


def get_task_queue() -> TaskQueue:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = TaskQueue()
        return _instance
