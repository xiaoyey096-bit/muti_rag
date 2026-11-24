import os

from dotenv import load_dotenv

# 项目根目录（由本文件位置动态推算，换机器/换盘符都不用改）形如 <你的路径>\shopkeeper_brain\knowledge
KNOWLEDGE_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 本地文件存储基础目录
LOCAL_BASE_DIR = os.path.join(KNOWLEDGE_ROOT, "temp_data")

# 前端页面静态资源目录   形如 <KNOWLEDGE_ROOT>\front
FRONT_PAGE_DIR = os.path.join(KNOWLEDGE_ROOT, "front")

# 导入流程的本地临时产物目录（MinerU 解析结果、chunks_*.json 等中间产物）
IMPORT_TEMP_DIR = os.path.join(KNOWLEDGE_ROOT, "processor", "import_processor", "temp_dir")

# 显式加载项目自己的 .env，保证在任何工作目录下都能读到 MINERU_BACKEND / MINERU_METHOD
load_dotenv(os.path.join(KNOWLEDGE_ROOT, ".env"))

def get_local_base_dir() -> str:
    """获取本地文件存储基础目录"""
    return LOCAL_BASE_DIR


def get_front_page_dir() -> str:
    """获取前端静态页面目录"""
    return FRONT_PAGE_DIR


def get_import_temp_dir() -> str:
    """获取导入流程的本地临时产物目录"""
    return IMPORT_TEMP_DIR


def get_mineru_subdir() -> str:
    """
    推算 MinerU 的输出子目录名，规则与
    processor/import_processor/nodes/pdf_to_md_node.py 的 _get_output_subdir 保持一致：
    pipeline -> <method>（例如 auto）；vlm-* -> vlm；hybrid-* -> hybrid_<method>
    """
    backend = os.getenv("MINERU_BACKEND", "pipeline")
    method = os.getenv("MINERU_METHOD", "auto")
    if backend.startswith("hybrid"):
        return f"hybrid_{method}"
    if backend.startswith("vlm"):
        return "vlm"
    return method


def get_mineru_md_path(doc_name: str) -> str:
    """MinerU 解析产物的 md 路径：<temp_dir>\<doc_name>\<subdir>\<doc_name>.md"""
    return os.path.join(IMPORT_TEMP_DIR, doc_name, get_mineru_subdir(), f"{doc_name}.md")


def get_mineru_new_md_path(doc_name: str) -> str:
    """md_to_img_node 处理后生成的备份文件：<mineru md 同目录>\<doc_name>_new.md"""
    return os.path.join(os.path.dirname(get_mineru_md_path(doc_name)), f"{doc_name}_new.md")
