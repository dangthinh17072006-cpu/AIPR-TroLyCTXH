"""Chuẩn hoá và tách từ tiếng Việt (bỏ dấu) dùng chung cho BM25 và embedding."""
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFD', text.lower().replace('đ', 'd'))
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')


def tokenize(text: str) -> list[str]:
    return re.findall(r'[a-z0-9]+', normalize(text))
