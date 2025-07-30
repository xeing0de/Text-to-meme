import struct
from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class WDBObject:
    path: str
    vector: list[float] = field(repr=False)  # Если нужен вывод вектора, убрать False
    is_text: bool | None = field(default=None)
    distance: float | None = field(default=None)

    @classmethod
    def from_base(cls, item: tuple[str, Tensor]):
        return WDBObject(
            path=item[0],
            vector=item[1].tolist(),
        )

    @classmethod
    def from_wdb(cls, obj) -> "WDBObject":
        return WDBObject(
            vector=obj.vector["default"],
            path=obj.properties["path"],
            is_text=obj.properties["is_text"],
            distance=obj.metadata.distance
        )

    def to_bytes(self) -> bytes:
        path_bytes = self.path.encode('utf-8')
        path_len = len(path_bytes)

        packed = struct.pack(f'>I{path_len}s384f', path_len, path_bytes, *self.vector)
        return packed

    @classmethod
    def from_bytes(cls, data: bytes) -> 'WDBObject':
        path_len = struct.unpack_from('>I', data)[0]

        offset = 4
        fmt = f'>{path_len}s384f'
        path_bytes, *vector = struct.unpack_from(fmt, data, offset)
        path = path_bytes.decode('utf-8')
        return cls(path=path, vector=vector)

