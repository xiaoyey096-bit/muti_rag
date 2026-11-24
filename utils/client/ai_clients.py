import os
import threading
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from openai import OpenAI
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from FlagEmbedding import FlagReranker
from muti_rag.utils.client.base import BaseClientManager, logger
from muti_rag.utils.env_selfcheck import ensure_transformers_safe_load

# 从 knowledge 目录加载 .env
_env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(_env_path)

# 清除 NO_PROXY 环境变量，避免 httpx 解析 IPv6 地址失败
os.environ.pop('NO_PROXY', None)
os.environ.pop('no_proxy', None)


class AIClients(BaseClientManager):
    """AI 模型类客户端"""

    _openai_client: Optional[OpenAI] = None
    _openai_lock = threading.Lock()

    _openai_llm_response_text_client: Optional[ChatOpenAI] = None
    _openai_llm_response_text_lock = threading.Lock()

    _openai_llm_response_json_client: Optional[ChatOpenAI] = None
    _openai_llm_response_json_lock = threading.Lock()

    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    _bge_m3_rerank_client: Optional[FlagReranker] = None
    _bge_m3_rerank_lock = threading.Lock()

    # ── VLM ──

    @classmethod
    def get_vlm_client(cls) -> OpenAI:
        return cls._get_or_create("_openai_client", cls._openai_lock, cls._create_vlm_client)

    @classmethod
    def _create_vlm_client(cls) -> OpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")

            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"OpenAI 客户端初始化成功 (base_url={base_url})")

            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    # ── LLM ──
    @classmethod
    def get_llm_client(cls, response_format: bool = True) -> ChatOpenAI:
        if response_format:
            return cls._get_or_create("_openai_llm_response_json_client", cls._openai_llm_response_json_lock,
                                      lambda: cls._create_llm_client(response_format))
        else:
            return cls._get_or_create("_openai_llm_response_text_client", cls._openai_llm_response_text_lock,
                                      lambda: cls._create_llm_client(response_format))

            # ── LLM ──

    @classmethod
    def _create_llm_client(cls, response_format) -> ChatOpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            model_name = cls._require_env('LLM_DEFAULT_MODEL')

            model_kwargs = {}
            if response_format:
                model_kwargs['response_format'] = {"type": "json_object"}

            llm_client = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                openai_api_key=api_key,
                openai_api_base=base_url,
                model_kwargs=model_kwargs
            )
            logger.info(f"OpenAI LLM 客户端初始化成功")
            return llm_client

        except EnvironmentError:
            raise
        except Exception as e:
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    # ── BGE-M3嵌入模型客户端 ──
    @classmethod
    def get_bge_m3_client(cls):
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3_client)

    @classmethod
    def _safe_bge_device(cls, requested: str) -> str:
        """
        显存预检：本机显卡同时被桌面应用/虚拟机占用，显存不足时按 .env 强上 GPU
        会在加载阶段把整个进程打崩（CUDA 分配失败甚至 access violation）。
        可用显存不足或 CUDA 不可用时回退 CPU。
        """
        if not requested.lower().startswith("cuda"):
            return requested
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA 不可用，BGE 回退到 CPU")
                return "cpu"
        except Exception as e:
            logger.warning(f"CUDA 自检失败，BGE 回退到 CPU：{e}")
            return "cpu"
        free_mb = cls._free_vram_mb()
        need_mb = int(os.getenv("BGE_MIN_FREE_VRAM_MB", "2500"))
        if free_mb is None or free_mb < need_mb:
            logger.warning(f"可用显存 {free_mb}MB < {need_mb}MB，BGE 回退到 CPU（显卡被其他进程占用）")
            return "cpu"
        return requested

    @staticmethod
    def _free_vram_mb() -> Optional[float]:
        """
        Windows WDDM 下 torch.cuda.mem_get_info 会把共享内存算作空闲（实测报 5.3GB
        而实际仅剩 1.6GB），必须用 nvidia-smi 的专用显存数值判断。
        """
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            return float(out.stdout.strip().splitlines()[0])
        except Exception:
            try:
                import torch
                return torch.cuda.mem_get_info()[0] / 1e6
            except Exception:
                return None

    @staticmethod
    def _limit_cpu_threads() -> None:
        # torch 多线程算子读取 transformers 的 mmap 权重在本机 Windows 上会 access violation
        import torch
        if torch.get_num_threads() > 1:
            torch.set_num_threads(1)
            logger.warning("BGE 以 CPU 推理运行：torch 线程数限制为 1（规避 mmap 权重崩溃）")

    @classmethod
    def _create_bge_m3_client(cls):
        """
        创建bge_m3 客户端
        Returns:
        """

        try:
            ensure_transformers_safe_load()
            # 1. 获取环境变量
            model_name = cls._require_env('BGE_M3_PATH')
            device = cls._safe_bge_device(cls._require_env('BGE_DEVICE'))
            fp16_str = cls._require_env('BGE_FP16')

            fp16 = fp16_str.lower() in ("true", "1")
            if device == "cpu":
                fp16 = False
                cls._limit_cpu_threads()
            # 2. 创建
            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=fp16
            )
            return bge_m3_ef
        except EnvironmentError as e:
            raise

        except Exception as e:
            raise ConnectionError(f"BGE_M3嵌入模型客户端创建失败: {e}") from e

    # ── BGE-M3重排序模型客户端 ──
    @classmethod
    def get_bge_m3_rerank_client(cls):
        return cls._get_or_create("_bge_m3_rerank_client",
                                  cls._bge_m3_rerank_lock,
                                  cls._create_bge_m3_rerank_client)

    @classmethod
    def _create_bge_m3_rerank_client(cls):
        """
        创建bge_m3 重排序模型客户端
        Returns:
        """

        try:
            ensure_transformers_safe_load()
            # 1. 获取环境变量
            model_name_or_path = cls._require_env('BGE_RERANKER_LARGE')
            device = cls._safe_bge_device(cls._require_env('BGE_DEVICE'))
            fp16_str = cls._require_env('BGE_FP16')
            fp16 = fp16_str.lower() in ("true", "1")
            if device == "cpu":
                fp16 = False
                cls._limit_cpu_threads()

            # 2. 创建
            reranker = FlagReranker(
                model_name_or_path=model_name_or_path,
                device=device,
                use_fp16=fp16
            )

            return reranker
        except EnvironmentError as e:
            raise

        except Exception as e:
            raise ConnectionError(f"BGE-M3重排序模型客户端创建失败: {e}") from e


if __name__ == '__main__':
    # llm_client: ChatOpenAI = AIClients.get_llm_client()
    #
    # llm_response = llm_client.invoke("请您给我讲一个笑话，要求输出格式是一个json")
    #
    # llm_result = llm_response.content
    #
    # import json
    #
    # result = json.loads(llm_result)
    #
    # print(result)
    #
    print(AIClients.get_bge_m3_rerank_client())
