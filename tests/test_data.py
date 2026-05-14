"""Tests for data utilities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from src.data import extract_metadata, clean_data


class TestExtractMetadata:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "id": [
                "12/2015/TT-BTC#1-Điều 1",
                "12/2015/TT-BTC#2-Điều 2",
                "45/2017/NĐ-CP#1-Điều 1",
                "01/2020/Luật-DN#1-Điều 1",
            ],
            "title": ["Điều 1", "Điều 2", "Điều 1", "Điều 1"],
            "text": ["Nội dung A " * 10, "Nội dung B " * 10, "Nội dung C " * 10, "Nội dung D " * 10],
        })

    def test_extracts_law_id(self, sample_df):
        result = extract_metadata(sample_df)
        assert result["law_id"].tolist() == [
            "12/2015/TT-BTC", "12/2015/TT-BTC", "45/2017/NĐ-CP", "01/2020/Luật-DN"
        ]

    def test_extracts_article_num(self, sample_df):
        result = extract_metadata(sample_df)
        assert result["article_num"].tolist() == [1, 2, 1, 1]

    def test_extracts_law_type(self, sample_df):
        result = extract_metadata(sample_df)
        # law_id "12/2015/TT-BTC" → split by "/" → ["12","2015","TT-BTC"]
        # parts[2] = "TT-BTC", split by "-" → ["TT","BTC"], last = "BTC"
        assert result["law_type"].tolist() == ["BTC", "BTC", "CP", "DN"]

    def test_extracts_year(self, sample_df):
        result = extract_metadata(sample_df)
        assert result["year"].tolist() == [2015, 2015, 2017, 2020]

    def test_no_hash_in_id(self):
        df = pd.DataFrame({
            "id": ["some_simple_id"],
            "title": ["Test"],
            "text": ["Test content " * 5],
        })
        result = extract_metadata(df)
        assert result["law_id"].iloc[0] == "some_simple_id"
        assert result["article_num"].iloc[0] == 0

    def test_preserves_original_columns(self, sample_df):
        result = extract_metadata(sample_df)
        for col in sample_df.columns:
            assert col in result.columns

    def test_year_fallback(self):
        df = pd.DataFrame({
            "id": ["law/invalid_year/type"],
            "title": ["Test"],
            "text": ["Test " * 5],
        })
        result = extract_metadata(df)
        assert result["year"].iloc[0] == 1999  # fallback


class TestCleanData:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "id": ["a", "b", "c", "d", "e"],
            "title": ["T1", "T2", "T3", "T4", "T5"],
            "text": [
                "Nội dung đủ dài " * 5,
                "Ngắn",  # too short
                "Nội dung C " * 5,
                "Nội dung C " * 5,  # duplicate of above
                "Nội dung E " * 5,
            ],
        })

    def test_removes_short_texts(self, sample_df):
        result = clean_data(sample_df, min_text_len=10)
        assert "b" not in result["id"].values

    def test_removes_duplicates(self, sample_df):
        result = clean_data(sample_df, min_text_len=10)
        # "Nội dung C " * 5 appears twice, one removed
        assert result["text"].value_counts().iloc[0] == 1

    def test_normalizes_unicode(self):
        df = pd.DataFrame({
            "id": ["x"],
            "title": ["Tiêu đề"],  # decomposed
            "text": ["Nội dung thử nghiệm " * 5],
        })
        result = clean_data(df, min_text_len=10)
        # NFC normalization
        assert result["title"].iloc[0] == "Tiêu đề"

    def test_preserves_count(self, sample_df):
        result = clean_data(sample_df, min_text_len=10)
        # 5 rows - 1 short - 1 duplicate = 3
        assert len(result) == 3
