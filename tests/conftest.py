import pytest
from src.config import Config


@pytest.fixture
def config():
    return Config.from_yaml("config/config.yaml")


@pytest.fixture
def cfg():
    return Config.from_yaml("config/config.yaml")
