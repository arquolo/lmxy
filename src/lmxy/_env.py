__all__ = ['env']

from pydantic_settings import BaseSettings


class _Env(BaseSettings):
    # HTTP
    SSL_VERIFY: bool = True
    RETRIES: int = 0
    # HF offine mode - for fastembed models
    HF_HUB_OFFLINE: bool = False
    TRANSFORMERS_OFFLINE: bool = False


env = _Env()
