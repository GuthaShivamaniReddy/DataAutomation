from pathlib import Path

import polars as pl
import pytest

from dataos.semantics.dictionary import SemanticDictionary

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "golden"
SEMANTIC_DICTIONARY_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "semantic_dictionary"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def orders_basic_df() -> pl.DataFrame:
    return pl.read_csv(FIXTURES_DIR / "orders_basic.csv")


@pytest.fixture
def tmp_storage_root(tmp_path: Path) -> Path:
    return tmp_path / "dataos_store"


@pytest.fixture
def sample_semantic_dictionary() -> SemanticDictionary:
    return SemanticDictionary.load_from_json(SEMANTIC_DICTIONARY_FIXTURES_DIR / "sample_dictionary.json")
