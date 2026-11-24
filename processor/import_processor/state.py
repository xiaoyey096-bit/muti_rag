"""

导入流程状态类型定义



定义完整的状态结构和辅助函数

"""

from typing import TypedDict, List

import copy


class ImportGraphState(TypedDict, total=False):


    """

    导入流程图状态



    包含整个导入流程中传递的所有数据

    """

    # ==================== 任务标识 ====================

    task_id: str  # 任务 ID，用于任务追踪(web交互的时候用到，实时看到节点的处理日志)

    # ==================== 控制标志 ====================

    is_md_read_enabled: bool  # 是否启用 MD 读取

    is_pdf_read_enabled: bool  # 是否启用 PDF 读取

    # ==================== 路径信息 ====================

    import_file_path: str  # 导入文件路径

    file_dir: str  # 导入(出)文件目录

    pdf_path: str  # PDF 文件路径

    md_path: str  # 转换后Markdown 文件路径

    # ==================== 文件信息 ====================

    file_title: str  # 文件标题（不含扩展名）

    item_name: str  # 识别出的商品/产品名称(方便程序员用)

    kb_id: str  # 所属知识库ID（多产品共用集合，按 kb_id 隔离）

    file_sha256: str  # 原始文件哈希（向量库按哈希精确清理旧切片）

    # ==================== 处理中间数据 ====================

    md_content: str  # Markdown 文档内容

    chunks: List  # 文档切片列表

    suggested_questions: List  # 导入期生成的用户常问3问（写入知识库注册表）

    # ==================== 默认状态 ====================




GRAPH_DEFAULT_STATE: ImportGraphState = {

    "task_id": "",

    "is_pdf_read_enabled": False,

    "is_md_read_enabled": False,

    "file_dir": "",

    "import_file_path": "",

    "pdf_path": "",

    "md_path": "",

    "file_title": "",

    "md_content": "",

    "chunks": [],

    "suggested_questions": [],

    "item_name": "",

    "kb_id": "",

    "file_sha256": "",

}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段

    Returns:
        新的状态实例

    Examples:
        >>> state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """
    获取默认状态副本

    Returns:
        状态副本（避免全局污染）
    """
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
