from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from pymilvus import MilvusClient, DataType
from muti_rag.processor.import_processor.base import BaseNode, setup_logging
from muti_rag.processor.import_processor.state import ImportGraphState
from muti_rag.processor.import_processor.exceptions import StateFieldError, ValidationError, MilvusError
from muti_rag.utils.client.storage_clients import StorageClients


@dataclass
class _SCALAR_FIELD_SPC:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


_SCALAR_FIELDS: [_SCALAR_FIELD_SPC] = (
    _SCALAR_FIELD_SPC(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="kb_id", datatype=DataType.VARCHAR, max_length=64),
)


class _MilvusSchemaBuilder():
    """
    负责处理和Milvus字段约束相关的逻辑
    """

    @staticmethod
    def build_schema(milvus_client: MilvusClient, dim: int):
        """
        创建schema
        Args:
            milvus_client:
            dim:

        Returns:
        enable_dynamic_field:动态字段：添加数据的时候可以提供在静态字段基础上额外的字段---静态字段（提前定义的）：

        """
        # 1. 创建schema
        schema = milvus_client.create_schema(enable_dynamic_field=True)

        # 2. 添加字段约束
        # 2.1 添加主键字段的约束 auto_id:1.能够自动生成值  2. 在一定的时间内有顺序 3.插入记录的时候无需提供该字段
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)

        # 2.2 添加向量字段的约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2.3 添加标量字段的约束[标量字段个数：5个]
        for spec in _SCALAR_FIELDS:
            kwargs: Dict = {
                "field_name": spec.field_name,
                "datatype": spec.datatype
            }
            if spec.max_length:
                kwargs['max_length'] = spec.max_length

            schema.add_field(**kwargs)

        return schema


class _MilvusInserter:

    def __init__(self, milvus_client: MilvusClient, collection_name: str):
        self._milvus_client = milvus_client
        self._collection_name = collection_name

    def insert_rows(self, data: List[Dict[str, Any]]):
        # 0. chunk_id 为自增主键：断点续跑时 chunk 里可能残留上次插入回填的 id，必须剔除
        for row in data:
            row.pop("chunk_id", None)

        # 1. 插入
        inserted_result = self._milvus_client.insert(collection_name=self._collection_name, data=data)

        # 2. 得到每一个chunk的id
        chunk_ids = inserted_result.get('ids')

        # 3. 回填到chunk中
        for chunk_id, chunk in zip(chunk_ids, data):
            chunk['chunk_id'] = chunk_id


class _MilvusIndexBuilder:

    @staticmethod
    def build_index_params(milvus_client: MilvusClient):
        index = milvus_client.prepare_index_params()

        # 稠密向量：AUTOINDEX
        index.add_index(field_name="dense_vector",
                        index_name="dense_vector_index",
                        index_type="AUTOINDEX",
                        metric_type="COSINE")

        # 稀疏向量：倒排索引
        index.add_index(field_name="sparse_vector",
                        index_name="sparse_vector_index",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP")

        return index


class ImportMilvusNode(BaseNode):
    """
    角色：充当门面（门面模式）
    """
    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args:
            state:

        Returns:

        """

        # 1. 校验state
        validated_chunks, dim = self._validate_state(state)
        kb_id = state.get('kb_id', '')
        file_sha256 = state.get('file_sha256', '') or ''

        # 2. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"MilVus客户端创建失败,异常原因{str(e)}")
            raise MilvusError(message=f"MilVus客户端创建失败,异常原因{str(e)}", node_name=self.name)

        # 3. 获取chunks集合
        chunks_collection = self.config.chunks_collection

        # 4. 创建集合
        self._create_chunks_collection(chunks_collection, milvus_client, dim)

        # 5. 幂等清理：同一篇文档重复导入时，先删掉它的旧 chunk
        #    （原实现只 insert 不 delete，重跑一次导入链向量库里就会多一份重复数据，
        #     检索时新旧两版切块会互相干扰）
        self._delete_stale_chunks(chunks_collection, milvus_client, validated_chunks, kb_id, file_sha256)

        # 6. 注入 kb_id（多产品隔离）与 file_sha256 后插入数据(静态字段都要提供)
        for chunk in validated_chunks:
            chunk['kb_id'] = kb_id
            if file_sha256:
                chunk['file_sha256'] = file_sha256
        _inserter = _MilvusInserter(milvus_client, chunks_collection)

        _inserter.insert_rows(validated_chunks)

        # 7. 返回state
        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], int]:

        self.log_step("validate", "参数校验")
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError("待入库的 chunks 为空或类型无效", self.name)

        validated_chunks = []
        for i, chunk in enumerate(chunks):
            # 类型不对 → 抛异常（和上游 embedding 节点保持一致）
            if not isinstance(chunk, dict):
                raise ValidationError(
                    f"chunks[{i}] 类型无效：期望 dict，实际为 {type(chunk).__name__}", self.name
                )
            # 缺少向量 → 跳过（嵌入可能部分失败，属于数据级容错）
            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                validated_chunks.append(chunk)
            else:
                self.logger.warning(f"chunks[{i}] 缺少混合向量，已跳过")

        if not validated_chunks:
            raise ValidationError("所有 chunk 均无有效向量，无法入库", self.name)

        dim = len(validated_chunks[0]["dense_vector"])
        self.logger.info(f"有效 chunks：{len(validated_chunks)}，向量维度：{dim}")
        return validated_chunks, dim

    def _delete_stale_chunks(self, chunks_collection: str, milvus_client: MilvusClient,
                             validated_chunks: List[Dict[str, Any]], kb_id: str = "",
                             file_sha256: str = ""):
        """
        幂等清理：按 kb_id + (file_title 或 file_sha256) 删除该文档已存在的旧 chunk。
        - file_sha256：同一文件重复/断点续跑导入时精确定位旧数据（含文件改名场景）
        - file_title：兼容 P0 之前导入的历史数据（它们没有 file_sha256 字段）
        Args:
            chunks_collection: chunk 集合名
            milvus_client: Milvus 客户端
            validated_chunks: 本次待入库的 chunk（从中取出 file_title）
            kb_id: 所属知识库ID（缺省时退化为仅按 file_title 清理）
            file_sha256: 原始文件哈希

        Returns:
            None
        """
        file_titles = {chunk.get("file_title") for chunk in validated_chunks if chunk.get("file_title")}
        if not file_titles and not file_sha256:
            self.logger.warning("chunks 中缺少 file_title 且无 file_sha256，跳过旧数据清理")
            return

        kb_expr = 'kb_id == "%s"' % str(kb_id).replace('"', '\\"') if kb_id else ""
        conditions = []
        for file_title in file_titles:
            ft = str(file_title).replace('"', '\\"')
            conditions.append('file_title == "%s"' % ft if not kb_id
                              else '(%s and file_title == "%s")' % (kb_expr, ft))
        if file_sha256:
            sha = str(file_sha256).replace('"', '\\"')
            conditions.append('file_sha256 == "%s"' % sha if not kb_id
                              else '(%s and file_sha256 == "%s")' % (kb_expr, sha))

        expr = " or ".join(conditions)
        stale = milvus_client.query(collection_name=chunks_collection, filter=expr,
                                    output_fields=["chunk_id"], limit=16384)
        if not stale:
            return
        stale_ids = [row["chunk_id"] for row in stale]
        milvus_client.delete(collection_name=chunks_collection, ids=stale_ids)
        self.logger.info(f"幂等清理：{chunks_collection} 中《{','.join(map(str, file_titles))}》"
                         f"旧 chunk {len(stale_ids)} 条已删除")

    def _create_chunks_collection(self, chunks_collection: str, milvus_client: MilvusClient, dim: int):

        # 1. 判断集合
        if milvus_client.has_collection(chunks_collection):
            self.logger.info(f"{chunks_collection}已存在 跳过创建")
            return
        # 2. 创建schema
        schema = _MilvusSchemaBuilder.build_schema(milvus_client, dim)

        # 3. 创建索引
        index_params = _MilvusIndexBuilder.build_index_params(milvus_client)

        # 4. 创建集合
        milvus_client.create_collection(collection_name=chunks_collection, schema=schema, index_params=index_params)


def _cli_main() -> None:
    import json
    from pathlib import Path
    from muti_rag.core.paths import get_import_temp_dir
    setup_logging()

    temp_dir = Path(get_import_temp_dir())
    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    state: ImportGraphState = {"chunks": content.get('chunks')}

    node = ImportMilvusNode()
    result_state = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"结果已保存至: {output_path}")


if __name__ == '__main__':
    _cli_main()
