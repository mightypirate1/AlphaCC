from fastapi import Response


class ServiceExceptionError(Exception):
    def __init__(self, source: Exception) -> None:
        self._source = source

    @property
    def source(self) -> Exception:
        return self._source

    def to_response(self) -> Response:
        return Response(
            content=str(self.source),
            status_code=400,
        )
