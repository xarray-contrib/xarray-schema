import json
from abc import abstractmethod
from typing import Any, Dict


class SchemaError(Exception):
    '''Custom Schema Error'''

    pass


class BaseSchema:

    _json_schema: Dict[str, Any]

    # def validate(self, obj: Any) -> None:
    #     pass

    @property
    @abstractmethod
    def json(self) -> Any:  # pragma: no cover
        pass

    def to_json(self, **dumps_kws) -> str:
        return json.dumps(self.json, **dumps_kws)
