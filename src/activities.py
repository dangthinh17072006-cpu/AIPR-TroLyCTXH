"""Đọc bảng mã hóa (mục II.2.a) trong data/CTXH.md và luật quy đổi điểm -> ngày (II.2.c)."""
import re
from functools import lru_cache
from pathlib import Path

from .models import Activity

DATA = Path(__file__).resolve().parent.parent / "data"


def to_number(text: str) -> float:
    return float(text.strip().replace(",", "."))


def to_range(text: str) -> tuple[float, float]:
    """'5 → 10' -> (5, 10); '5' -> (5, 5)."""
    parts = [p for p in text.split("→") if p.strip()]
    return to_number(parts[0]), to_number(parts[-1])


def parse_table(md: str) -> list[Activity]:
    activities = []
    category = ""
    for line in md.splitlines():
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 5 or not re.fullmatch(r"\d{6}", cells[1]):
            continue
        category = cells[0] or category  # ô "Lĩnh vực" trống thì lấy của dòng trên
        p_min, p_max = to_range(cells[3])
        d_min, d_max = to_range(cells[4])
        # ngày phải bằng điểm / 10, lệch thì báo lỗi luôn
        if abs(p_min / 10 - d_min) > 1e-9 or abs(p_max / 10 - d_max) > 1e-9:
            raise ValueError(f"Bảng lệch điểm/ngày ở mã {cells[1]}")
        activities.append(Activity(cells[1], category, cells[2], p_min, p_max, d_min, d_max))
    return activities


@lru_cache(maxsize=1)
def load_activities() -> dict[str, Activity]:
    md = (DATA / "CTXH.md").read_text(encoding="utf-8")
    return {a.code: a for a in parse_table(md)}


def fmt(x: float) -> str:
    """0.5 -> '0,5'; 2.0 -> '2'."""
    return f"{x:g}".replace(".", ",")


def fmt_range(low: float, high: float) -> str:
    return fmt(low) if low == high else f"{fmt(low)}→{fmt(high)}"


def points_to_days(total_points: float) -> float:
    """Mục II.2.c: chia 10; phần dư dưới 5 điểm bỏ, đúng 5 điểm tính 0,5 ngày, trên 5 điểm tính 1 ngày."""
    if total_points < 0:
        raise ValueError("điểm không được âm")
    days, rest = divmod(round(total_points, 6), 10)
    if rest == 5:
        return days + 0.5
    if rest > 5:
        return days + 1
    return days


def required_days(program_years: int) -> int:
    """Mục I.3: số ngày tối thiểu theo độ dài khóa (4 năm trở lên: 4 ngày, dưới 4 năm: 2 ngày)."""
    return 4 if program_years >= 4 else 2
