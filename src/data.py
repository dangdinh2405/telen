"""Shared data utilities used by TELEN modules."""
import unicodedata
import pandas as pd


def load_raw_data(parquet_path: str) -> pd.DataFrame:
    """Load the raw parquet file."""
    return pd.read_parquet(parquet_path)


def extract_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extract law_id, article_num, law_type, year from id column."""
    df = df.copy()

    def parse_id(id_str):
        if "#" in id_str:
            parts = id_str.split("#")
            law_id = parts[0]
            article_part = parts[1]
            article_num = int(article_part.split("-")[0])
        else:
            law_id = id_str
            article_num = 0
        return law_id, article_num

    parsed = df["id"].apply(parse_id)
    df["law_id"] = parsed.apply(lambda x: x[0])
    df["article_num"] = parsed.apply(lambda x: x[1])

    def extract_law_type(law_id):
        parts = law_id.split("/")
        if len(parts) >= 3:
            return parts[2].split("-")[-1] if "-" in parts[2] else parts[2]
        return "unknown"

    df["law_type"] = df["law_id"].apply(extract_law_type)

    def extract_year(law_id):
        parts = law_id.split("/")
        if len(parts) >= 2:
            year_str = parts[1]
            try:
                year = int(year_str)
                return year if year >= 100 else year + 1900
            except ValueError:
                pass
        return 1999

    df["year"] = df["law_id"].apply(extract_year)
    return df


def clean_data(df: pd.DataFrame, min_text_len: int = 10) -> pd.DataFrame:
    """Remove short/empty texts and duplicates."""
    df = df.copy()
    df = df[df["text"].str.len() >= min_text_len].reset_index(drop=True)
    df["title"] = df["title"].apply(lambda x: unicodedata.normalize("NFC", str(x)))
    df["text"] = df["text"].apply(lambda x: unicodedata.normalize("NFC", str(x)))
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    return df
