from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


CVS_COLUMNS = [
    "video", "critical_view", "initial_minute", "initial_second", "final_minute",
    "final_second", "two_structures", "cystic_plate", "hepatocystic_triangle", "Total",
]


def load_phase_starts(path: str | Path) -> dict[str, float]:
    starts: dict[str, float] = {}
    with Path(path).open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            frame, phase = line.rstrip().split("\t")
            starts.setdefault(phase, int(frame) / 25.0)
    return starts


def _xlsx_rows(path: str | Path) -> list[list[str]]:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in root.findall(".//x:sheetData/x:row", ns):
        values = []
        for cell in row.findall("x:c", ns):
            inline = cell.find("x:is/x:t", ns)
            value = cell.find("x:v", ns)
            values.append(inline.text if inline is not None else value.text if value is not None else "")
        rows.append(values)
    return rows


def load_cvs_intervals(path: str | Path, video_id: int) -> dict[str, list[tuple[float, float, int]]]:
    rows = _xlsx_rows(path)
    header = rows[0]
    index = {name: header.index(name) for name in CVS_COLUMNS}
    result = {"two_structures": [], "cystic_plate": [], "hepatocystic_triangle": []}
    for row in rows[1:]:
        if not row or int(row[index["video"]]) != int(video_id):
            continue
        start = 60 * float(row[index["initial_minute"]]) + float(row[index["initial_second"]])
        end = 60 * float(row[index["final_minute"]]) + float(row[index["final_second"]])
        if end <= start:
            continue
        for key in result:
            result[key].append((start, end, int(row[index[key]])))
    return result
