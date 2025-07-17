# Управление БД

## Добавить в БД объект
```python
from database import WDBObject
from database import WeaviateDB

db = WeaviateDB()

v = [0.0] * 384         # Вектор, вычисленный заранее

obj = WDBObject(
    path="/test/path",  # Путь к картинке
    is_text=False,      # Есть ли текст на картинке
    distance=None,      # Тут не используется
    vector=v
)

db.add_object(obj)      # Добавление
```

## Получить ближайшие объекты по вектору
```py
from pprint import pprint

from database import WeaviateDB

db = WeaviateDB()

v = [0.1] * 384  # Вектор, по которому нужно найти ближайшие значения

result = db.near_objects(v, limit=5)
pprint(result)
```
Пример вывода:
```text
[WDBObject(path='path1', is_text=None, distance=0.0),
 WDBObject(path='path1', is_text=None, distance=0.0),
 WDBObject(path='/test/path', is_text=False, distance=1.0),
 WDBObject(path='/test/path', is_text=False, distance=1.0),
 WDBObject(path='/test/path', is_text=False, distance=1.0)]
```