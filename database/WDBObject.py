from dataclasses import dataclass, field


@dataclass
class WDBObject:
    path: str
    is_text: bool
    distance: float | None
    vector: list[float] = field(repr=False)  # Если нужен вывод вектора, убрать False

    @classmethod
    def from_wdb(cls, obj) -> "WDBObject":
        return WDBObject(
            vector=obj.vector["default"],
            path=obj.properties["path"],
            is_text=obj.properties["is_text"],
            distance=obj.metadata.distance
        )
