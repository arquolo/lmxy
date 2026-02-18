__all__ = ['env']

from pydantic_settings import BaseSettings


class _Env(BaseSettings):
    # HTTP
    SSL_VERIFY: bool = True
    RETRIES: int = 0
    MAX_CONNECTIONS: int = 100
    MAX_KEEP_ALIVE_CONNECTIONS: int | None = 20
    KEEP_ALIVE_TIMEOUT: float = 15
    # HF offine mode - for fastembed models
    HF_HUB_OFFLINE: bool = False
    TRANSFORMERS_OFFLINE: bool = False


env = _Env()
