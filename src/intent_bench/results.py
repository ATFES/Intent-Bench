from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import uuid


def record(value):
    return asdict(value) if is_dataclass(value) else value


def save(results, directory=None, *, command=None):
    if directory is None:
        directory = Path("runs") / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = [record(r) for r in results]
    temporary=directory/('.results-'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(), "command": command,
        "results": rows}, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory/'results.json')
    fields = ["workload", "case", "implementation", "status", "latency_ms", "throughput", "unit",
              "device", "method", "scope", "cache"]
    with (directory / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {key: row.get(key) for key in fields}
            values["latency_ms"] = row.get("median_ms")
            values["throughput"], values["unit"] = throughput(row)
            values["cache"] = row.get("metadata", {}).get("cache_policy", "unspecified")
            writer.writerow(values)
    write_html(rows, directory / "report.html", source=directory / "results.json")
    return directory


def read(path):
    path = Path(path)
    if path.suffix=='.csv':
        from .history import import_csv
        return [record(row) for row in import_csv(path)]
    data = json.loads((path / "results.json" if path.is_dir() else path).read_text())
    return data["results"] if isinstance(data, dict) else data


def throughput(row):
    metrics = row.get("metrics", {})
    for name, unit in (("tflops", "TFLOP/s"), ("gbps", "GB/s")):
        if name in metrics:
            return metrics[name], unit
    return None, ""


def _number(value):
    return "—" if value is None else f"{value:.6g}"


def _groups(results):
    groups = {}
    for value in results:
        row = record(value)
        key = (row.get("device", ""), row.get("method", ""), row.get("scope", ""),
               row.get("metadata", {}).get("cache_policy", "unspecified"))
        groups.setdefault(key, []).append(row)
    return groups


def _cells(row, with_throughput):
    values = [f"{row['workload']}/{row['case']}", row["implementation"], row["status"], _number(row.get("median_ms"))]
    if with_throughput:
        value, unit = throughput(row)
        values.append(f"{_number(value)} {unit}".rstrip())
    return values


def table(results):
    lines = []
    for key, rows in _groups(results).items():
        with_throughput = any(throughput(row)[0] is not None for row in rows)
        lines.append(" / ".join(part for part in key if part))
        lines.append("workload / case | implementation | status | latency ms" + (" | throughput" if with_throughput else ""))
        for row in rows:
            lines.append(" | ".join(_cells(row, with_throughput)))
            if row.get("error"):
                lines.append("  " + row["error"].splitlines()[0][:250])
    return "\n".join(lines)


def comparison_table(changes):
    lines = ["workload / case | implementation | before ms | after ms | change"]
    for row in changes:
        delta = f"{(1 / row['speedup'] - 1) * 100:+.2f}%" if row["speedup"] else row["change"]
        lines.append(f"{row['workload']}/{row['case']} | {row['implementation']} | {_number(row['before_ms'])} | {_number(row['after_ms'])} | {delta}")
    return "\n".join(lines)


def compare(before, after):
    def key(r):
        return (r["workload"], r["case"], r["implementation"], r["device"], r["method"], r["scope"],
                json.dumps(r["parameters"], sort_keys=True), r.get("metadata", {}).get("machine"),
                r.get("metadata", {}).get("threads"), r.get("metadata", {}).get("cache_policy", "unspecified"),
                r.get("metadata", {}).get("cache_bytes"))
    left, right = {}, {}
    for source, target in ((before, left), (after, right)):
        for item in source:
            row = record(item)
            identity = key(row)
            if identity in target:
                raise ValueError("multiple measurements for the same comparison key; select one run")
            target[identity] = row
    changes = []
    for identity in dict.fromkeys([*left, *right]):
        old, new = left.get(identity), right.get(identity)
        valid = bool(old and new and old["correctness"] == new["correctness"] == "passed"
                     and old["median_ms"] and new["median_ms"])
        changes.append({"workload": identity[0], "case": identity[1], "implementation": identity[2],
                        "change": "added" if old is None else "removed" if new is None else "measured",
                        "before_ms": old.get("median_ms") if old else None,
                        "after_ms": new.get("median_ms") if new else None,
                        "before_status": old.get("status") if old else None,
                        "after_status": new.get("status") if new else None,
                        "speedup": old["median_ms"] / new["median_ms"] if valid else None})
    return sorted(changes, key=lambda r: (r["speedup"] is None, r["speedup"] or 0))


def write_html(results, path, *, source=None):
    esc = lambda value: html.escape(str(value))
    sections = []
    for key, rows in _groups(results).items():
        with_throughput = any(throughput(row)[0] is not None for row in rows)
        columns = ["Workload / case", "Implementation", "Status", "Latency (ms)"]
        if with_throughput:
            columns.append("Throughput")
        body = []
        for row in rows:
            cells = _cells(row, with_throughput)
            if row.get("error"):
                cells[2] = esc(cells[2]) + "<details><summary>Error</summary><pre>" + esc(row["error"]) + "</pre></details>"
                rendered = [esc(value) if index != 2 else value for index, value in enumerate(cells)]
            else:
                rendered = [esc(value) for value in cells]
            body.append("<tr>" + "".join("<td>" + value + "</td>" for value in rendered) + "</tr>")
        sections.append("<section><h2>" + esc(" / ".join(part for part in key if part)) +
                        "</h2><table><thead><tr>" + "".join("<th>" + esc(c) + "</th>" for c in columns) +
                        "</tr></thead><tbody>" + "".join(body) + "</tbody></table></section>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    source_link = ""
    if source is not None:
        from urllib.parse import quote
        reference = quote(os.path.relpath(Path(source).resolve(), Path(path).resolve().parent))
        source_link = '<p>Source records: <a href="' + esc(reference) + '">' + esc(Path(source).name) + '</a></p>'
    Path(path).write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Intent Bench</title>
<style>body{font:14px system-ui;margin:32px;color:#182331}h1{font-size:26px}h2{font-size:14px;font-weight:500;color:#526173}section{margin:24px 0;overflow:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}th{background:#f4f6f8}pre{white-space:pre-wrap;max-width:600px}input{padding:8px;width:320px}</style>
<h1>Operator latency</h1><p>Median latency of the complete call. Throughput is optional. Failed or unchecked rows are diagnostic measurements.</p>
<input id="filter" placeholder="Filter workload, implementation or status">
''' + source_link + "".join(sections) + '''<script>document.querySelector('#filter').oninput=e=>{for(const r of document.querySelectorAll('tbody tr'))r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase())}</script></html>''')
