import os.path
import hashlib
import logging
import time
import uuid
from datetime import datetime
from fastapi import UploadFile
from muti_rag.core.paths import get_local_base_dir
from muti_rag.processor.import_processor.exceptions import FileProcessingError
from muti_rag.utils.kb_registry_util import ensure_kb
from muti_rag.utils.task_util import add_running_task, add_done_task, add_node_duration, \
    create_task, save_task_artifacts
from muti_rag.utils.upload_session_util import register_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UpLoadService:

    def get_base_dir(self) -> str:
        # %Y%m%d:年月日
        # %Y:四位 %y:两位
        return os.path.join(get_local_base_dir(), datetime.now().strftime("%Y%m%d"))

    """
    处理文件上传相关的逻辑
    """

    def process_upload_file(self, file: UploadFile, kb_id: str = "", product_name: str = ""):
        """
        处理文件上传

        1. 将上传的文件存储到本地临时目录(主要为了做中转)
        2. 将上传的文件存储到远程minio(主要持久化)
        3. 解析/归属知识库 kb_id（已有则追加，新产品则新建，缺省用文件名兜底）
        4. 将file_dir / import_file_path / task_id / kb_id 返回
        Args:
            file:
            kb_id: 目标知识库ID（向已有产品追加文档时提供）
            product_name: 产品名（新建知识库时提供）

        Returns:

        """

        # 1. 生成任务id

        task_id = str(uuid.uuid4().hex[:8])  # 真正的随机 获取前8个随机数
        add_running_task(task_id, "upload_file")
        start_time = time.time()

        # 2. 生成日期目录并且将日期目录和临时目录拼接到一起
        base_file_dir = self.get_base_dir()

        # 3. 构建文档完整归属目录
        file_dir = os.path.join(base_file_dir, task_id)

        # 3.5 规范化文件名（还原 Windows curl 上传时中文名被按 latin-1 解码产生的乱码）
        filename = self._normalize_filename(file.filename)

        # 3.6 归属知识库：已有 kb_id 则追加；否则按 product_name 新建；都缺省用文件名兜底（provisional）
        file_title = os.path.splitext(filename)[0]
        kb_id = ensure_kb(kb_id, product_name, file_title)

        # 4. 保存文件到临时目录（同时计算 sha256）
        import_file_path, file_sha256 = self.save_upload_file_to_local(file, file_dir, filename)

        # 4.1 登记到 files 集合（file_id=sha256），后续同内容上传可直接秒传
        file_info = register_file(import_file_path, filename, file_sha256)
        file_id = file_info["file_id"]

        # 4.2 创建持久化任务文档（队列调度的基础）
        create_task(task_id, kb_id=kb_id, file_title=file_title,
                    import_file_path=import_file_path, file_dir=file_dir,
                    file_sha256=file_sha256, file_id=file_id)

        end_time = time.time()
        add_done_task(task_id, "upload_file")
        add_node_duration(task_id, "upload_file", end_time - start_time)
        save_task_artifacts(task_id, {"file_dir": file_dir, "file_title": file_title})

        # 5. 返回任务信息（file_id 供去重索引与前端记录）
        return task_id, import_file_path, file_dir, kb_id, file_sha256, file_id

    @staticmethod
    def _normalize_filename(filename: str) -> str:
        """
        Windows 下用 curl(Git Bash) 上传中文名文件时，文件名可能以 GBK 字节发出，
        被 multipart 解析器按 latin-1 解码成乱码（如 "用户手册" -> "ÓÃ»§ÊÖ²á"）。
        当文件名只含 latin-1 字符、且能按 GBK 还原出中文时，还原为正确名称。
        """
        if not filename:
            return filename
        try:
            raw = filename.encode('latin-1', errors='strict')
        except UnicodeEncodeError:
            # 含非 latin-1 字符，说明本来就是正常的 Unicode 文件名，无需处理
            return filename
        for enc in ('gbk', 'gb18030'):
            try:
                decoded = raw.decode(enc)
            except UnicodeDecodeError:
                continue
            # 还原结果必须含中文，避免误伤正常的西欧语言文件名
            if decoded != filename and any('\u4e00' <= ch <= '\u9fff' for ch in decoded):
                logger.info(f"已还原文件名乱码: {filename} -> {decoded}")
                return decoded
        return filename

    def save_upload_file_to_local(self, file: UploadFile, file_dir: str, filename: str):
        """
        保存文件到临时目录，同时计算 sha256（用于幂等清理与任务恢复）
        Args:
            file: 文件上传对象
            file_dir: 上传文件的目录
            filename: 规范化后的文件名

        Returns:
            (导入文件路径, sha256)
        """
        # 1. 创建文件的归属目录
        os.makedirs(file_dir, exist_ok=True)

        # 2. 构建导入文件的路径
        import_file_path = os.path.join(file_dir, filename)

        # 3. 分批写入并同步计算哈希（大文件不整读进内存）
        hasher = hashlib.sha256()
        try:
            with open(import_file_path, "wb") as f:
                while True:
                    buf = file.file.read(1024 * 1024)
                    if not buf:
                        break
                    hasher.update(buf)
                    f.write(buf)
        except IOError as e:
            logger.info(f"{filename}写入临时目录失败 原因:{str(e)}")
            raise FileProcessingError(message=f"{filename}写入临时目录失败 原因:{str(e)}")

        # 4. 返回导入的文件路径与哈希
        return import_file_path, hasher.hexdigest()
