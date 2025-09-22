__all__ = ['env']

from pydantic_settings import BaseSettings


class _Env(BaseSettings):
    # HTTP
    SSL_VERIFY: bool = True
    RETRIES: int = 0
    MAX_CONNECTIONS: int | None = None
    MAX_KEEP_ALIVE_CONNECTIONS: int | None = 20
    # HF offine mode - for fastembed models
    HF_HUB_OFFLINE: bool = False
    TRANSFORMERS_OFFLINE: bool = False


env = _Env()
