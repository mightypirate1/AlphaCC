from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PredictRequest(_message.Message):
    __slots__ = ("request_id", "state_tensor", "moves", "model_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_TENSOR_FIELD_NUMBER: _ClassVar[int]
    MOVES_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    state_tensor: bytes
    moves: bytes
    model_id: int
    def __init__(self, request_id: _Optional[int] = ..., state_tensor: _Optional[bytes] = ..., moves: _Optional[bytes] = ..., model_id: _Optional[int] = ...) -> None: ...

class PredictResponse(_message.Message):
    __slots__ = ("request_id", "pi_logits", "wdl_logits")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PI_LOGITS_FIELD_NUMBER: _ClassVar[int]
    WDL_LOGITS_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    pi_logits: bytes
    wdl_logits: bytes
    def __init__(self, request_id: _Optional[int] = ..., pi_logits: _Optional[bytes] = ..., wdl_logits: _Optional[bytes] = ...) -> None: ...

class ServerInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ChannelInfo(_message.Message):
    __slots__ = ("channel_id", "model_loaded", "model_version")
    CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_LOADED_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    channel_id: int
    model_loaded: bool
    model_version: int
    def __init__(self, channel_id: _Optional[int] = ..., model_loaded: bool = ..., model_version: _Optional[int] = ...) -> None: ...

class ServerInfoResponse(_message.Message):
    __slots__ = ("game", "channels", "batch_sizes", "static_mode")
    GAME_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZES_FIELD_NUMBER: _ClassVar[int]
    STATIC_MODE_FIELD_NUMBER: _ClassVar[int]
    game: str
    channels: _containers.RepeatedCompositeFieldContainer[ChannelInfo]
    batch_sizes: _containers.RepeatedScalarFieldContainer[int]
    static_mode: bool
    def __init__(self, game: _Optional[str] = ..., channels: _Optional[_Iterable[_Union[ChannelInfo, _Mapping]]] = ..., batch_sizes: _Optional[_Iterable[int]] = ..., static_mode: bool = ...) -> None: ...

class LoadModelRequest(_message.Message):
    __slots__ = ("channel_id", "onnx_bytes", "version")
    CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    ONNX_BYTES_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    channel_id: int
    onnx_bytes: bytes
    version: int
    def __init__(self, channel_id: _Optional[int] = ..., onnx_bytes: _Optional[bytes] = ..., version: _Optional[int] = ...) -> None: ...

class LoadModelResponse(_message.Message):
    __slots__ = ("success", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error: str
    def __init__(self, success: bool = ..., error: _Optional[str] = ...) -> None: ...
