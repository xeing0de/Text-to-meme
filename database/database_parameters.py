from weaviate.collections.classes.config import (
    CollectionConfig,
    PropertyConfig,
    DataType,
    Tokenization,
    VectorIndexConfigHNSW,
    VectorDistances,
    InvertedIndexConfig,
    ShardingConfig,
    ReplicationConfig,
    VectorIndexType,
    StopwordsConfig, BM25Config, MultiTenancyConfig,
    ReplicationDeletionStrategy, StopwordsPreset
)
from weaviate.collections.classes.config_vector_index import VectorFilterStrategy

W_NAME = "Meme"

W_CONFIG = CollectionConfig(
    name=W_NAME,
    description="Мемы",
    properties=[
        PropertyConfig(
            name="path",
            description="Путь к изображению",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD,
            index_filterable=False,
            index_searchable=False,
            index_range_filters=False,
            nested_properties=None,
            vectorizer=None,
            vectorizer_config=None,
            vectorizer_configs=None
        ),
        PropertyConfig(
            name="is_text",
            description="Есть ли текст на картинке",
            data_type=DataType.BOOL,
            tokenization=None,
            index_filterable=True,
            index_searchable=False,
            index_range_filters=False,
            nested_properties=None,
            vectorizer=None,
            vectorizer_config=None,
            vectorizer_configs=None
        )
    ],
    vector_index_config=VectorIndexConfigHNSW(
        distance_metric=VectorDistances.COSINE,
        ef=128,
        ef_construction=128,
        max_connections=64,
        cleanup_interval_seconds=300,
        flat_search_cutoff=10000,
        dynamic_ef_min=100,
        dynamic_ef_max=500,
        dynamic_ef_factor=8,
        filter_strategy=VectorFilterStrategy.ACORN,
        skip=False,
        vector_cache_max_objects=100000,
        multi_vector=None,
        quantizer=None
    ),
    vector_index_type=VectorIndexType.HNSW,
    inverted_index_config=InvertedIndexConfig(
        bm25=BM25Config(
            k1=1.2,  # стандартное значение
            b=0.75  # стандартное значение
        ),
        cleanup_interval_seconds=60,
        index_null_state=False,
        index_property_length=False,
        index_timestamps=True,
        stopwords=StopwordsConfig(
            preset=StopwordsPreset.NONE,  # Можно: "none", "en", "de", "ru", ...
            additions=[],  # Дополнительные стоп-слова
            removals=[]  # Удалить из стандартного набора
        )
    ),
    sharding_config=ShardingConfig(
        virtual_per_physical=128,  # Кол-во виртуальных шардов на физический
        desired_count=1,  # Сколько всего физических шардов хотим
        actual_count=1,  # Сколько реально будет (обычно = desired)
        desired_virtual_count=128,  # Виртуальных шардов всего
        actual_virtual_count=128,  # То же, но как будет применено
        key="_id",  # Поле, по которому шардится
        strategy="hash",  # Алгоритм шардинга: hash / range
        function="murmur3"  # Хэш-функция: murmur3 (обычно дефолт)
    ),
    replication_config=ReplicationConfig(
        factor=1,
        async_enabled=False,
        deletion_strategy=ReplicationDeletionStrategy.DELETE_ON_CONFLICT
    ),
    generative_config=None,
    multi_tenancy_config=MultiTenancyConfig(
        enabled=False,
        auto_tenant_creation=False,
        auto_tenant_activation=False
    ),
    references=[],  # обычно пустой список
    reranker_config=None,
    vectorizer_config=None,
    vector_config=None,
    vectorizer="none"
)
