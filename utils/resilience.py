"""
韧性组件（P2）：错误分类、退避、依赖级熔断、并发闸门、结构化日志。

设计取舍：
- 熔断按依赖（DashScope / BGE / Milvus / MinIO / Mongo / GPU-MinerU）聚合，
  而不是按任务聚合——外部依赖故障时所有任务一起降速，而不是各自把重试次数耗尽。
- 单机进程内实现即可；多 Worker 时每个进程各有一份熔断状态，语义仍是"本地降级"。
"""

import json
import logging
import random
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ==================== 错误分类 ====================

TRANSIENT = "TRANSIENT"
RESOURCE = "RESOURCE"
PERMANENT = "PERMANENT"

_PERMANENT_EXC = (
    FileNotFoundError, PermissionError, IsADirectoryError, UnicodeDecodeError, ValueError,
)
_PERMANENT_NAMES = ("StateFieldError", "ValidationError", "ConfigurationError")
_TRANSIENT_HINTS = (
    "timeout", "timed out", "connection", "reset by peer", "refused",
    "temporarily", "unavailable", "429", "502", "503", "504",
    "requesttimetooskewed", "clock", "broken pipe", "eof occurred",
)
_RESOURCE_HINTS = (
    "out of memory", "cuda", "oom", "no space left", "disk full",
    "显存", "内存不足", "磁盘",
)


def _root_cause(exc: BaseException) -> BaseException:
    seen = set()
    cur = exc
    while getattr(cur, "cause", None) is not None and id(cur) not in seen:
        seen.add(id(cur))
        cur = cur.cause
    if isinstance(cur, BaseException):
        return cur
    return exc


def classify_error(exc: BaseException) -> str:
    """把异常归类为 TRANSIENT / RESOURCE / PERMANENT。"""
    root = _root_cause(exc)
    name = type(root).__name__
    text = f"{root}".lower()

    if name in _PERMANENT_NAMES or isinstance(root, _PERMANENT_EXC):
        # 文件缺失/权限/参数校验类，重试无意义
        if name == "ValueError" and any(h in text for h in _TRANSIENT_HINTS):
            return TRANSIENT
        return PERMANENT
    if isinstance(root, MemoryError) or any(h in text for h in _RESOURCE_HINTS):
        return RESOURCE
    if isinstance(root, (ConnectionError, TimeoutError, OSError)):
        return TRANSIENT
    return TRANSIENT


# ==================== 重试参数 ====================

STEP_MAX_ATTEMPTS: Dict[str, int] = {
    "entry_node": 1,
    "pdf_to_md_node": 3,
    "md_to_img_node": 3,
    "document_split_node": 1,
    "item_name_recognition_node": 3,
    "embedding_chunk_node": 4,
    "import_milvus_node": 3,
    "suggest_questions_node": 2,
}
# RESOURCE 类额外允许 1 次降级重试
RESOURCE_EXTRA_ATTEMPTS = 1
# 任务级总尝试上限 + 连续无进展失败上限
TASK_MAX_ATTEMPTS = int(__import__("os").getenv("TASK_MAX_ATTEMPTS", "15"))
TASK_MAX_NO_PROGRESS = 3

# 步骤 → 依赖（用于熔断）
STEP_DEPENDENCIES: Dict[str, List[str]] = {
    "entry_node": [],
    "pdf_to_md_node": ["gpu_mineru"],
    "md_to_img_node": ["dashscope", "minio"],
    "document_split_node": [],
    "item_name_recognition_node": ["dashscope"],
    "embedding_chunk_node": ["bge"],
    "import_milvus_node": ["milvus"],
    "suggest_questions_node": ["dashscope"],
}

BACKOFF_BASE = 5.0
BACKOFF_CAP = 300.0


def backoff_delay(attempt: int) -> float:
    """5·2ⁿ + 抖动，封顶 5 分钟。"""
    delay = min(BACKOFF_BASE * (2 ** max(0, attempt - 1)), BACKOFF_CAP)
    return round(delay + random.uniform(0, 2), 2)


# ==================== 熔断器 ====================

class CircuitBreaker:

    def __init__(self, name: str, window: float = 60.0, min_samples: int = 10,
                 failure_rate: float = 0.5, consecutive_limit: int = 5,
                 base_open: float = 60.0, max_open: float = 600.0):
        self.name = name
        self.window = window
        self.min_samples = min_samples
        self.failure_rate = failure_rate
        self.consecutive_limit = consecutive_limit
        self.base_open = base_open
        self.max_open = max_open

        self._lock = threading.Lock()
        self._events: deque = deque()  # (ts, ok)
        self._state = "closed"
        self._open_until = 0.0
        self._open_duration = base_open
        self._probe_inflight = False
        self._opened_at = 0.0
        self._last_error = ""

    # --- 内部 ---
    def _prune(self, now: float):
        while self._events and now - self._events[0][0] > self.window:
            self._events.popleft()

    def _trip(self, now: float, reason: str):
        self._state = "open"
        self._open_until = now + self._open_duration
        self._opened_at = now
        self._last_error = reason
        logger.warning(f"熔断器 [{self.name}] OPEN {self._open_duration:.0f}s（{reason}）")

    def allow(self) -> bool:
        now = time.time()
        with self._lock:
            if self._state == "open":
                if now < self._open_until:
                    return False
                self._state = "half_open"
                self._probe_inflight = True
                logger.info(f"熔断器 [{self.name}] HALF-OPEN，放行 1 个探测请求")
                return True
            if self._state == "half_open":
                if self._probe_inflight:
                    return False
                self._probe_inflight = True
                return True
            return True

    def release_probe(self):
        with self._lock:
            self._probe_inflight = False

    def on_success(self):
        now = time.time()
        with self._lock:
            self._probe_inflight = False
            if self._state in ("open", "half_open"):
                self._state = "closed"
                self._open_duration = self.base_open
                self._events.clear()
                logger.info(f"熔断器 [{self.name}] CLOSED（探测成功）")
                return
            self._prune(now)
            self._events.append((now, True))

    def on_failure(self, error: str = ""):
        now = time.time()
        with self._lock:
            self._probe_inflight = False
            if self._state == "half_open":
                self._open_duration = min(self._open_duration * 2, self.max_open)
                self._trip(now, error or "半开探测失败")
                return
            if self._state == "open":
                return
            self._prune(now)
            self._events.append((now, False))
            consecutive = 0
            for _, ok in reversed(self._events):
                if ok:
                    break
                consecutive += 1
            total = len(self._events)
            failures = sum(1 for _, ok in self._events if not ok)
            rate = failures / total if total else 0.0
            if consecutive >= self.consecutive_limit or (total >= self.min_samples and rate >= self.failure_rate):
                self._open_duration = self.base_open
                self._trip(now, error or f"连续失败 {consecutive} 次 / 失败率 {rate:.0%}")

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            total = len(self._events)
            failures = sum(1 for _, ok in self._events if not ok)
            cooldown = max(0.0, self._open_until - now) if self._state == "open" else 0.0
            return {
                "name": self.name,
                "state": self._state,
                "samples": total,
                "failures": failures,
                "failure_rate": round(failures / total, 3) if total else 0.0,
                "cooldown_seconds": round(cooldown, 1),
                "open_duration": self._open_duration,
                "last_error": self._last_error[:200],
            }


class BreakerRegistry:

    def __init__(self):
        self._lock = threading.Lock()
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name)
            return self._breakers[name]

    def allow_step(self, step: str) -> Tuple[bool, str]:
        """检查该步骤的依赖是否都处于可放行状态；返回 (允许, 阻塞的依赖名)。"""
        for dep in STEP_DEPENDENCIES.get(step, []):
            if not self.get(dep).allow():
                return False, dep
        return True, ""

    def release_step(self, step: str):
        for dep in STEP_DEPENDENCIES.get(step, []):
            self.get(dep).release_probe()

    def record_step(self, step: str, ok: bool, error: str = ""):
        for dep in STEP_DEPENDENCIES.get(step, []):
            breaker = self.get(dep)
            if ok:
                breaker.on_success()
            else:
                breaker.on_failure(error)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            names = list(self._breakers.keys())
        return [self.get(n).snapshot() for n in names]


RESILIENCE = BreakerRegistry()

# ==================== 并发闸门 ====================

_GPU_SEM = threading.Semaphore(int(__import__("os").getenv("GPU_PARSE_CONCURRENCY", "1")))
_IO_SEM = threading.Semaphore(int(__import__("os").getenv("HEAVY_IO_CONCURRENCY", "2")))

# pdf 解析全局串行（GPU/内存），embedding/milvus 允许并发 2
STEP_SEMAPHORES = {
    "pdf_to_md_node": _GPU_SEM,
    "embedding_chunk_node": _IO_SEM,
    "import_milvus_node": _IO_SEM,
}


def step_semaphore(step: str) -> Optional[threading.Semaphore]:
    return STEP_SEMAPHORES.get(step)


# ==================== 结构化日志 ====================

def slog(event: str, level: int = logging.INFO, **fields) -> None:
    """JSON 行日志：一次任务一个 trace_id，可按字段检索。"""
    payload = {"event": event, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), **fields}
    logger.log(level, json.dumps(payload, ensure_ascii=False, default=str))


def user_message(error_class: str) -> str:
    return {
        TRANSIENT: "网络或服务暂时不稳定，系统正在自动重试",
        RESOURCE: "服务器资源紧张，稍后会自动继续",
        PERMANENT: "文件解析失败，请检查文件是否完整或格式是否支持",
    }.get(error_class, "处理过程中出现异常")
