from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """文件上传响应 —— POST /upload 返回"""
    message: str = Field(..., description="响应消息")
    task_id: str = Field(..., description="任务ID")
    kb_id: str = Field("", description="所属知识库ID")


class KbCreateRequest(BaseModel):
    """新建知识库 —— POST /kbs 请求体"""
    product_name: str = Field(..., description="产品名（如 DGX Spark / RS-12 数字万用表）")
    display_name: str = Field("", description="界面展示名，缺省用 product_name")
    hint: str = Field("", description="示例问题/引导提示")


class KbUpdateRequest(BaseModel):
    """更新知识库展示信息 —— PATCH /kbs/{kb_id} 请求体"""
    display_name: Optional[str] = None
    hint: Optional[str] = None
    product_name: Optional[str] = None


class KbResponse(BaseModel):
    """知识库信息 —— GET /kbs 列表元素"""
    kb_id: str
    product_name: str = ""
    display_name: str = ""
    hint: str = ""
    doc_titles: List[str] = Field(default_factory=list)
    item_names: List[str] = Field(default_factory=list)
    suggested_questions: List[str] = Field(default_factory=list, description="导入期生成的用户常问3问")
    provisional: bool = False
    created_at: float = 0
    updated_at: float = 0


class TaskStatusResponse(BaseModel):
    """任务状态响应 —— GET /status/{task_id} 返回"""
    status: str = Field(..., description="任务状态")
    done_list: List[str] = Field(..., description="已完成节点列表")
    running_list: List[str] = Field(..., description="正在运行节点列表")
    durations: Dict[str, float] = Field(default=dict, description="各节点耗时(秒)")
    error: str = Field("", description="失败原因（status=failed 时）")
    resumable: bool = Field(False, description="失败任务是否支持断点续跑")
    resume_count: int = Field(0, description="已续跑次数")
    current_step: str = Field("", description="当前/最近执行的步骤")
    error_class: str = Field("", description="错误分类 TRANSIENT/RESOURCE/PERMANENT")
    attempts: Dict[str, int] = Field(default_factory=dict, description="各步骤已消耗的尝试次数")
    next_retry_in: int = Field(0, description="waiting_retry 状态下距离自动重试的秒数")


class UploadInitRequest(BaseModel):
    """分片上传初始化 —— POST /uploads/init 请求体"""
    filename: str = Field(..., description="原始文件名")
    size: int = Field(..., description="文件字节数")
    sha256: str = Field("", description="整文件 sha256（提供后可秒传/断点续传）")
    chunk_size: int = Field(0, description="分片大小（字节），缺省 5MB")
    kb_id: str = Field("", description="目标知识库 ID（可空）")
    product_name: str = Field("", description="新建知识库时的产品名（可空）")


class TaskCreateRequest(BaseModel):
    """文件转导入任务 —— POST /files/{file_id}/tasks 请求体"""
    kb_id: str = Field("", description="目标知识库 ID（可空）")
    product_name: str = Field("", description="新建知识库时的产品名（可空）")
