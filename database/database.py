import weaviate
from weaviate.collections import Collection

from .WDBObject import WDBObject
from .database_parameters import W_NAME, W_CONFIG


class WeaviateDB:
    def __init__(self):  # Можно добавить порты и прочие. Если захочется, конечно
        self.name = W_NAME
        self.config = W_CONFIG

        self.client = weaviate.connect_to_local(port=4654, grpc_port=50051)
        self.collection: Collection = ...
        self.check_schema()

    def check_schema(self):
        if not self.client.collections.exists(self.name):
            self.client.collections.create_from_config(self.config)
            print("Схема создана")
        print("Схема уже есть")
        self.collection = self.client.collections.get(self.name)

    def add_object(self, wdb_object: WDBObject, /) -> None:
        self.collection.data.insert(
            properties={
                "path": wdb_object.path,
                "is_text": wdb_object.is_text,
            },
            vector=wdb_object.vector
        )

    def near_objects(self, vector: list[float], /, limit: int = 5) -> list[WDBObject]:
        result = self.collection.query.near_vector(
            near_vector=vector,
            limit=limit,
            include_vector=True,
            return_metadata=["distance"]
        ).objects
        return list(map(WDBObject.from_wdb, result))

    def __del__(self):
        self.client.close()
