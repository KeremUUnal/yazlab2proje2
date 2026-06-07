"""
Pytest fixture'lari — tum test dosyalari tarafindan kullanilir.
"""
import pytest
from src.config import Config


@pytest.fixture
def cfg():
    """Merkezi config fixture'i."""
    return Config.from_yaml("config/config.yaml")
