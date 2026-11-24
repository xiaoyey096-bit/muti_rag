# 1. 内置包
import subprocess, time, json, sys
import re
import shutil
import threading
import unicodedata
from typing import Optional, Tuple
from pathlib import Path
# 2. 三方包
# 3. 自己的包

from muti_rag.processor.import_processor.base import BaseNode, setup_logging
from muti_rag.processor.import_processor.state import ImportGraphState
from muti_rag.processor.import_processor.exceptions import StateFieldError, PdfConversionError

# MinerU 的 pipeline 后端会把整份 PDF 的所有页面渲染进内存，
# 同一进程内串行执行，避免多个导入任务并发解析叠加内存导致 OOM
_MINERU_LOCK = threading.Lock()


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点的处理逻辑入口
        :param state:
        :return:
        """

        # 核心逻辑（接收pdf的文件path 利用mineru解析工具将pdf解析成md）

        # 1. 获取导入文件的路径以及输出目录
        import_file_path_obj, file_dir_obj = self._validate_state(state)

        # 2. 执行mineru解析；页数超过阈值时分片解析再合并，避免一次性载入全部页面导致内存耗尽
        page_count = self._get_page_count(import_file_path_obj)
        if page_count > self.config.mineru_chunk_threshold:
            self._parse_in_chunks(import_file_path_obj, file_dir_obj, page_count)
        else:
            # 命令： mineru -p input_path -o output_dir -b pipeline -d cuda:0 --source=local
            processed_code = self._execute_mineru_parse(import_file_path_obj, file_dir_obj)
            if processed_code != 0:
                raise PdfConversionError(message="MinerU解析PDF失败", node_name=self.name)

        # 3. 获取解析后md_path
        md_path = self._get_md_path(import_file_path_obj, file_dir_obj)

        # 3.1 恢复标题里的章节号（源 PDF 部分数字字体 ToUnicode 缺失，MinerU 会把 2.1 解析成 \x00.\x00）
        self._recover_heading_numbers(md_path, str(import_file_path_obj))

        # 3.2 清理 MinerU 可能残留的空字节（否则 md 会被识别成二进制文件，且污染后续切分）
        self._clean_md_file(md_path)

        # 4. 更新state["md_path"]
        state['md_path'] = md_path

        # 5. 返回state
        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """

        :param state: 导入图谱节点状态
        :return: 导入文件的路径以及输出目录
        Tuple[Path,Path]
        """

        self.log_step("step1", "准备校验和获取解析文件路径和输出目录")

        # 1. 获取解析的文件path
        import_file_path = state.get('import_file_path', '')

        # 2. 判断是否为空
        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name='import_file_path', expected_type=str)

        # 3. 标准化解析文件的路径
        import_file_path_obj = Path(import_file_path)

        # 4. 判断是否是一个有效的路径
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='import_file_path', expected_type=str,
                                  message="解析文件的路径不存在")

        # 5. 获取输出文件目录
        file_dir = state.get('file_dir', '')

        # 6. 判断输出文件目录
        if not file_dir:
            # 6.1 获取导入文件的目录（默认值）
            file_dir = import_file_path_obj.parent

        # 7. 标准化输出目录
        file_dir_obj = Path(file_dir)

        # 8. 判断是否是一个有效的目录
        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='file_dir', expected_type=str,
                                  message="输出目录不存在")

        self.logger.info(f"解析的文件路径{import_file_path}")
        self.logger.info(f"输出的文件目录{file_dir}")

        # 9. 返回校验通过的
        return import_file_path_obj, file_dir_obj

    def _execute_mineru_parse(self, import_file_path_obj: Path,
                              file_dir_obj: Path,
                              start_page: Optional[int] = None,
                              end_page: Optional[int] = None) -> int:
        """

        :param import_file_path_obj: 解析文件的path路径
        :param file_dir_obj:  解析后的文件输出目录
        :param start_page: 起始页（0起，含），None 表示整份
        :param end_page: 结束页（0起，含），None 表示整份
        :return:  状态[0或者非0]
        0:成功的（底层封装状态码）
        非0：失败的
        """
        # <python> -m mineru.cli.client -p input -o output -b pipeline -d cuda:0 -m auto --source local

        # 1. 定义cmd
        cmd = self._build_mineru_cmd(import_file_path_obj, file_dir_obj, start_page, end_page)

        # 2. 利用子进程执行cmd命令(子进程解析的日志【正常日志和错误日志都要】)
        # 子进程（执行命令产生的日志[正常、错误日志]）-----管子----外部线程（_execute_mineru_parse）
        start_time = time.time()
        with _MINERU_LOCK:
            proc = subprocess.Popen(args=cmd,
                                    stdout=subprocess.PIPE,  # 接收正确日志
                                    stderr=subprocess.STDOUT,  # 接收错误日志
                                    text=True,  # 输出二进制字节流，输出字符串
                                    errors="replace",  # 特殊的字符码替换成?、菱形
                                    encoding="utf-8",  # utf-8进行解码
                                    bufsize=1  # 实时输出。按行输出遇到\n换行符 就将日志产生出来
                                    )

            # 3. 实时打印日志
            for line in proc.stdout:
                self.logger.info(f"MinerU解析产生的日志：{line}")

            processed_result = proc.wait()
        # 4. 主线程等待子进程做完（状态码0：成功 反之失败）
        end_time = time.time()

        # 4.1 mineru运行期异常时仍可能返回0，用产物是否落盘做兜底校验
        if processed_result == 0:
            expect_md_path = Path(self._get_md_path(import_file_path_obj, file_dir_obj))
            if not expect_md_path.exists():
                self.logger.error(f"MinerU返回状态码0，但未生成预期的md：{expect_md_path}")
                processed_result = 1

        if processed_result == 0:
            self.logger.info(f"MinerU解析PDF成功 耗时:{end_time - start_time:.2f}s")
        else:
            self.logger.info("MinerU解析PDF失败")

        return processed_result

    def _build_mineru_cmd(self, import_file_path_obj: Path, file_dir_obj: Path,
                          start_page: Optional[int] = None,
                          end_page: Optional[int] = None) -> list:
        """
        组装 mineru 命令行
        不走 PATH 上的 mineru.exe：该启动器内嵌的是安装时的 python 绝对路径，
        当 venv 位于含中文的目录时，fasttext 无法打开 lid.176.ftz（非 ASCII 路径）。
        改为显式用 MINERU_PYTHON 指定的解释器以模块方式启动。
        :return: 命令参数列表
        """
        config = self.config
        python_exe = config.mineru_python or sys.executable

        if not config.mineru_python and not python_exe.isascii():
            self.logger.warning(
                "未配置MINERU_PYTHON，且当前解释器路径含非ASCII字符："
                f"{python_exe}。MinerU将在语言检测处崩溃"
                "（fasttext无法打开lid.176.ftz），请在.env中把MINERU_PYTHON指向ASCII路径的python.exe"
            )

        cmd = [
            python_exe,
            "-m",
            "mineru.cli.client",
            "-p",
            str(import_file_path_obj),
            "-o",
            str(file_dir_obj),
            "-b",
            config.mineru_backend,
            "-d",
            config.mineru_device,
            "-m",
            config.mineru_method,
            "--source",
            config.mineru_source
        ]
        if start_page is not None:
            cmd += ["-s", str(start_page)]
        if end_page is not None:
            cmd += ["-e", str(end_page)]
        self.logger.info(f"MinerU实际执行命令：{subprocess.list2cmdline(cmd)}")

        return cmd

    def _get_page_count(self, pdf_path_obj: Path) -> int:
        """读取PDF总页数；失败时返回0（按整份解析处理，保持原行为）"""
        try:
            import pypdf
            return len(pypdf.PdfReader(str(pdf_path_obj)).pages)
        except Exception as e:
            self.logger.warning(f"读取PDF页数失败，按整份解析处理: {e}")
            return 0

    def _parse_in_chunks(self, pdf_path_obj: Path, file_dir_obj: Path, page_count: int) -> None:
        """
        大PDF分片解析：MinerU 的 pipeline 后端会一次性把整份PDF的所有页面渲染进内存，
        页数多（尤其扫描件）时容易耗尽系统内存。这里按页范围多次调用 MinerU，
        每次只载入一个分片，最后把各分片的 md 与 images 合并成标准产物。
        """
        config = self.config
        chunk_size = max(1, config.mineru_chunk_pages)
        ranges = [(start, min(start + chunk_size - 1, page_count - 1))
                  for start in range(0, page_count, chunk_size)]
        self.logger.info(
            f"PDF共{page_count}页，超过分片阈值{config.mineru_chunk_threshold}页，"
            f"分{len(ranges)}片解析（每片≤{chunk_size}页）")

        chunk_root = file_dir_obj / f"{pdf_path_obj.stem}_chunks"
        if chunk_root.exists():
            shutil.rmtree(chunk_root)

        merged_texts = []
        image_dirs = []
        try:
            for idx, (start, end) in enumerate(ranges, 1):
                chunk_out_dir = chunk_root / f"part{idx}"
                chunk_out_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"分片{idx}/{len(ranges)}：第{start + 1}-{end + 1}页")

                code = self._execute_mineru_parse(pdf_path_obj, chunk_out_dir, start, end)
                part_md_path = Path(self._get_md_path(pdf_path_obj, chunk_out_dir))
                if code != 0 or not part_md_path.exists():
                    raise PdfConversionError(
                        message=f"第{idx}片（第{start + 1}-{end + 1}页）MinerU解析失败",
                        node_name=self.name)

                merged_texts.append(part_md_path.read_text(encoding="utf-8"))
                part_img_dir = part_md_path.parent / "images"
                if part_img_dir.exists():
                    image_dirs.append(part_img_dir)

            final_md_path = Path(self._get_md_path(pdf_path_obj, file_dir_obj))
            final_md_path.parent.mkdir(parents=True, exist_ok=True)
            final_md_path.write_text("\n\n".join(merged_texts), encoding="utf-8")
            self._merge_images(image_dirs, final_md_path.parent / "images")
            self.logger.info(f"分片解析完成，已合并为 {final_md_path}")
        finally:
            shutil.rmtree(chunk_root, ignore_errors=True)

    def _merge_images(self, src_dirs: list, dst_dir: Path) -> None:
        """把各分片的图片复制到合并后的 images 目录；MinerU 图片名按内容哈希生成，同名即同图"""
        if not src_dirs:
            return
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src_dir in src_dirs:
            for img in src_dir.iterdir():
                if not img.is_file():
                    continue
                target = dst_dir / img.name
                if target.exists():
                    continue
                shutil.copy2(img, target)
                copied += 1
        self.logger.info(f"已合并分片图片{copied}张到 {dst_dir}")

    @staticmethod
    def _norm_title(s: str) -> str:
        """标题归一化：NFKC 归一化（全角转半角、兼容字转正体）后去掉所有空白。"""
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))

    def _extract_pdf_heading_numbers(self, pdf_path: str) -> dict:
        """
        用 pypdf 直接抽取 PDF 文本层里形如「2.1 标题」「2.3.4.2 标题」的编号标题，
        得到 {归一化标题: 编号}。pypdf 对这些数字字体的 ToUnicode 解析是正确的，
        而 MinerU 的解析会把这些数字读成 \x00，因此这里以 pypdf 为基准恢复。
        """
        try:
            import pypdf
        except ImportError:
            self.logger.warning("未安装 pypdf，跳过标题编号恢复")
            return {}

        number_by_title = {}
        try:
            reader = pypdf.PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text() or ""
                for line in text.splitlines():
                    m = re.match(r"^(\d+(?:\.\d+)+)\s+(.+)$", line.strip())
                    if m:
                        number_by_title[self._norm_title(m.group(2))] = m.group(1)
        except Exception as e:
            self.logger.warning(f"pypdf 抽取标题编号失败: {e}")
            return {}
        return number_by_title

    def _recover_heading_numbers(self, md_path: str, pdf_path: str) -> None:
        """
        标题里被 MinerU 解析成 \x00 的章节号（如 # \x00.\x00 DGX Spark 发行说明）恢复成
        正确的编号（# 2.1 DGX Spark 发行说明）。仅处理 # 开头的标题行，正文里的数字不动。
        """
        md_path_obj = Path(md_path)
        try:
            raw = md_path_obj.read_text(encoding="utf-8")
        except OSError as e:
            self.logger.warning(f"恢复标题编号前读取失败: {e}")
            return
        # 没有空字节说明无需恢复，跳过全本 PDF 的文本抽取（扫描件大PDF上很耗时）
        if "\x00" not in raw:
            return

        number_by_title = self._extract_pdf_heading_numbers(pdf_path)
        if not number_by_title:
            return

        lines = raw.split("\n")
        changed = 0
        for i, line in enumerate(lines):
            if "\x00" not in line or not line.lstrip().startswith("#"):
                continue
            # 去掉 "# " 前缀后，剥离前面的空字节/点/空白，剩下的是标题文本
            body = line.lstrip()[1:].strip()
            title = re.sub(r"^[\x00\.\s]+", "", body)
            norm_title = self._norm_title(title)
            number = number_by_title.get(norm_title)
            if number:
                lines[i] = f"# {number} {title}"
                changed += 1

        if changed:
            md_path_obj.write_text("\n".join(lines), encoding="utf-8")
            self.logger.info(f"已恢复 {changed} 个标题的章节编号")

    def _clean_md_file(self, md_path: str) -> None:
        """
        MinerU 会把标题里的数字（如 2.1）解析成 \x00.\x00 的形态，
        空字节会让 md 被当成二进制文件且污染下游切分。这里读回产物并清理：
        1) 「\x00.」后面紧跟非数字/非空字节的孤立组合整组删掉（如「过\x00.滤」「\x00. 条款」），
           但保留「\x00.\x00」数字序列里的点（标题已在 _recover_heading_numbers 处理，剩余如版本号 v\x00.\x00\x00.\x00）；
        2) 删除其余空字节；
        3) 清理行首残留的孤立点号。
        """
        md_path_obj = Path(md_path)
        try:
            raw = md_path_obj.read_text(encoding="utf-8")
        except OSError as e:
            self.logger.warning(f"清理md空字节前读取失败: {e}")
            return
        if "\x00" not in raw:
            return
        cleaned = re.sub(r"\x00\.(?![\x00\d])", "", raw)
        cleaned = cleaned.replace("\x00", "")
        cleaned = re.sub(r"^\.\s+", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^(#\s*)\.\s+", r"\1", cleaned, flags=re.MULTILINE)
        md_path_obj.write_text(cleaned, encoding="utf-8")
        self.logger.info(f"已清理 {md_path_obj.name} 中的空字节")

    def _get_output_subdir(self) -> str:
        """
        mineru 不同 backend 的输出子目录命名规则（见 mineru/cli/common.py: prepare_env）
        pipeline          -> <method>        例如 auto
        vlm-*             -> vlm
        hybrid-*          -> hybrid_<method> 例如 hybrid_auto
        :return: 输出子目录名
        """
        backend = self.config.mineru_backend
        method = self.config.mineru_method
        if backend.startswith("hybrid"):
            return f"hybrid_{method}"
        if backend.startswith("vlm"):
            return "vlm"
        return method

    def _get_md_path(self, import_file_path_obj: Path, file_dir_obj: Path) -> str:
        """
        获取解析后md的路径
        :param import_file_path_obj:
        :param file_dir_obj:
        :return:
        md_path= <file_dir>\万用表的使用\auto\万用表的使用.md

        Path:吉祥三包：name:全名[文件名字.后缀]   stem【文件名字，没有后缀】    suffix【文件的后缀】
        """

        file_name = import_file_path_obj.stem

        return str(file_dir_obj / file_name / self._get_output_subdir() / f"{file_name}.md")


##########################################
# 测试
##########################################

if __name__ == '__main__':
    setup_logging()
    # 1. 构建节点实例
    pdf_to_md_node = PdfToMdNode()

    # 2. 构建该节点状态
    # 默认解析 temp_dir 下的万用表的使用.pdf，也可通过命令行参数指定：
    #   python -m knowledge.processor.import_processor.nodes.pdf_to_md_node <pdf路径> [输出目录]
    temp_dir_obj = Path(__file__).resolve().parents[1] / "temp_dir"
    init_state = {
        "import_file_path": sys.argv[1] if len(sys.argv) > 1 else str(temp_dir_obj / "万用表的使用.pdf"),
        "file_dir": sys.argv[2] if len(sys.argv) > 2 else str(temp_dir_obj)
    }

    # 3. 直接调用process
    result = pdf_to_md_node.process(init_state)

    # 4. 序列化（将对象转成字符串） 反序列化（将字符串转成对象）
    result_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(result_str)
