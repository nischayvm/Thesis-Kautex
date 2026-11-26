"""Export saved ANNAlyzer results to one filterable Excel workbook per version."""

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from urllib.parse import quote

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.series_factory import SeriesFactory
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
MODELS = {"patchcore": "PatchCore", "anomalydino": "AnomalyDINO", "unet": "U-Net", "flownomaly": "Flownomaly"}
METRICS = (("pixel_auroc", "Pixel AUROC"), ("pixel_ap", "Pixel AP"), ("aupro", "AUPRO"),
           ("image_auroc", "Image AUROC"), ("image_ap", "Image AP"))
SIZES = (("small_lt64", "Small <64"), ("medium_64_to_511", "Medium 64–511"), ("large_ge512", "Large ≥512"))
STAGES = ("validation", "test")
SOURCES = {"report.json", "training.json", "history.json", "smoke.json", "complete.json"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def numeric_fields(value, prefix=""):
    """Retain unrecognized numeric fields with their original JSON location."""
    if numeric(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_fields(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from numeric_fields(child, f"{prefix}[{index}]")


@dataclass
class Result:
    model: str
    run: Path
    source: Path
    stage: str
    light: str
    data: dict
    config: dict
    variant: str = ""

    @property
    def id(self):
        return self.model, self.run.name, self.stage, self.light, self.variant

    @property
    def normal(self):
        return self.data.get("normal_cohort", self.data.get("varied_normal", {}))

    @property
    def defects(self):
        return self.data.get("defect_cohort", self.data.get("fixed_pose_defects", {}))

    @property
    def metrics(self):
        return self.data.get("metrics", {**self.data, **self.defects})

    @property
    def images(self):
        return self.data.get("rows", self.data.get("predictions", []))

    @property
    def stages(self):
        stages = {**self.data.get("fit_measurements", {}), **self.data.get("measurements", {})}
        for epoch in self.data.get("history", []):
            stages.update({f"epoch_{epoch['epoch']}_{name}": timing
                           for name, timing in epoch.get("measurements", {}).items()})
        return stages

    @property
    def cohort(self):
        frames = sorted((row.get("id", row.get("sample_id", "")), row.get("cohort", row.get("role", "")))
                        for row in self.images)
        evidence = (self.data.get("dataset_sha256") or self.config.get("split_sha256"), frames,
                    self.data.get("pixel_scope"))
        if not frames:
            evidence = (*evidence, self.id)
        return hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()[:12]


def primary_reports(run):
    if (run / "report.json").is_file():
        yield run / "report.json"
    for stage in STAGES:
        direct = run / stage / "report.json"
        if direct.is_file():
            yield direct
    marker = run / "complete.json"
    if marker.is_file():
        for light in read_json(marker).get("lights", []):
            if light not in ("red", "green", "regular"):
                continue
            for stage in STAGES:
                report = run / light / stage / "report.json"
                if report.is_file():
                    yield report


def results_from(model, run, source, data):
    config = data.get("configuration", {})
    if isinstance(data.get("lighting"), dict):
        for light, result in data["lighting"].items():
            if isinstance(result, dict) and isinstance(result.get("metrics"), dict):
                yield Result(model, run, source, "validation", light, result, config)
    elif isinstance(data.get("metrics"), dict) and isinstance(data.get("banks"), dict):
        for variant, metrics in data["metrics"].items():
            budget = variant.split("_", 1)[0]
            result = {"metrics": metrics, "predictions": data.get("per_image", {}).get(budget, []),
                      "measurements": data.get("measurements", {}), "checkpoint": data["banks"].get(budget, {})}
            yield Result(model, run, source, "validation", config.get("fit_colour", "red"), result, config, variant)
    elif data.get("stage") in STAGES and ("defect_cohort" in data or "fixed_pose_defects" in data):
        if data["stage"] == "test" and not config:
            validation = source.parent.parent / "validation/report.json"
            if validation.is_file():
                config = read_json(validation).get("configuration", {})
        yield Result(model, run, source, data["stage"], data.get("lighting", "regular"), data, config)


def discover(root):
    versions = defaultdict(list)
    for folder in sorted((root / "Anamoly_detection").glob("*/results/V*")):
        if not folder.is_dir() or not re.fullmatch(r"V[1-9]\d*", folder.name):
            continue
        label = MODELS.get(folder.parents[1].name, folder.parents[1].name)
        for run in sorted(folder.iterdir()):
            if not run.is_dir():
                continue
            reports = [(source, read_json(source)) for source in primary_reports(run)]
            if reports:
                versions[folder.name].append((label, run, reports))
    return dict(sorted(versions.items(), key=lambda entry: int(entry[0][1:])))


def json_sources(run):
    for file in sorted(run.rglob("*.json")):
        if file.name in SOURCES and not any(part.startswith(("test_pending_", ".")) for part in file.relative_to(run).parts):
            yield file


def peak(stages, *keys):
    values = [row[key] for row in stages.values() if isinstance(row, dict) for key in keys if numeric(row.get(key))]
    return max(values, default=None)


def result_row(record):
    metrics, data, config = record.metrics, record.data, record.config
    normals, defects = record.normal, record.defects
    normal_count, defect_count = normals.get("frames"), defects.get("frames")
    if normal_count is None and record.images:
        normal_count = sum(frame.get("role") == "normal" for frame in record.images)
        defect_count = len(record.images) - normal_count
    false_positive, true_positive = normals.get("false_positives"), defects.get("true_positives")
    timings = [row.get("inference_seconds", row.get("inference_s")) for row in record.images]
    times = [value for value in timings if numeric(value)]
    checkpoint = data.get("checkpoint", {})
    bank = config.get("bank_dimensions") or checkpoint.get("dimensions") or (None, None)
    sizes = metrics.get("size_specific", {})
    return (*record.id, *(metrics.get(key) for key, _ in METRICS), data.get("threshold"),
            normal_count, defect_count, normal_count - false_positive if false_positive is not None else None,
            false_positive, true_positive, defect_count - true_positive if true_positive is not None else None,
            defects.get("sensitivity"), normals.get("false_positive_rate"),
            metrics.get("mean_inference_s", mean(times) if times else None),
            peak(record.stages, "rss_peak_mib", "rss_peak_mib_sampled"),
            peak(record.stages, "cuda_peak_allocated_mib"), peak(record.stages, "cuda_peak_reserved_mib"),
            *bank, config.get("bank_mib", checkpoint.get("bank_mib")),
            *(sizes.get(key, {}).get("aupro") for key, _ in SIZES),
            data.get("protocol", config.get("protocol")), record.cohort, record.source)


ID = ("Model", "Run", "Stage", "Lighting", "Variant")
HEADERS = (*ID, *(label for _, label in METRICS), "Threshold", "Normal frames", "Defective frames",
           "TN", "FP", "TP", "FN", "Sensitivity", "False-positive rate", "Inference s/frame", "Peak RAM MiB",
           "Peak VRAM allocated MiB", "Peak VRAM reserved MiB", "Bank rows", "Bank dimensions", "Bank MiB",
           *(f"{label} AUPRO" for _, label in SIZES), "Protocol", "Cohort ID", "Source JSON")


def link_path(path, output):
    try:
        return quote(Path(os.path.relpath(path, output)).as_posix(), safe="/")
    except ValueError:
        return path.resolve().as_uri()


def sheet(book, name, columns, rows, output):
    page = book.create_sheet(name)
    page.append(columns)
    page.freeze_panes = "B2"
    for cell in page[1]:
        cell.fill = PatternFill("solid", fgColor="164B44")
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    page.row_dimensions[1].height = 31
    for index, row in enumerate(rows, 2):
        page.append([str(item) if isinstance(item, Path) else item for item in row])
        for column, item in enumerate(row, 1):
            cell = page.cell(index, column)
            if isinstance(item, Path):
                cell.hyperlink = link_path(item, output)
                cell.value = item.name
                cell.style = "Hyperlink"
            elif isinstance(item, str):
                cell.data_type = "s"
            elif isinstance(item, float):
                cell.number_format = "0.00%" if columns[cell.column - 1] in ("Sensitivity", "False-positive rate") else "0.0000"
    page.auto_filter.ref = page.dimensions
    for index, title in enumerate(columns, 1):
        width = 38 if title in ("Run", "Protocol", "JSON field", "Path") else max(13, min(len(title) + 3, 27))
        page.column_dimensions[get_column_letter(index)].width = width
    return page


def latest_results(records):
    latest = {}
    for item in records:
        if item.variant:
            continue
        key = item.stage, item.light, item.model
        stamp = re.search(r"\d{8}T\d{6}_\d{6}Z$", item.run.name)
        rank = (stamp.group() if stamp else "", item.source.stat().st_mtime_ns)
        if key not in latest or rank > latest[key][0]:
            latest[key] = rank, item
    return sorted((item for _, item in latest.values()), key=lambda item: (item.stage, item.light, item.cohort, item.model))


def add_charts(book, summary, latest):
    charts = book.create_sheet("Charts", 2)
    charts["A1"] = "Latest completed results by lighting and stage"
    charts["A1"].font = Font(size=16, bold=True, color="164B44")
    charts["A2"] = "Cohorts and training protocols are recorded in Latest results. Empty metrics are unavailable."
    blocks = defaultdict(list)
    for idx, item in enumerate(latest, 2):
        blocks[item.stage, item.light, item.cohort].append(idx)
    position = 4
    total = 0
    score_names = [label for _, label in METRICS]
    sections = (("Detection and localization", score_names, "Score", 1),
                ("Defect size AUPRO", [f"{label} AUPRO" for _, label in SIZES], "AUPRO", 1),
                ("Inference", ["Inference s/frame"], "Seconds per frame", None),
                ("CUDA memory", ["Peak VRAM allocated MiB", "Peak VRAM reserved MiB"], "MiB", None))
    for (stage, light, cohort), rows in blocks.items():
        row_start, row_end = rows[0], rows[-1]
        plotted = 0
        for title, names, unit, limit in sections:
            columns = [HEADERS.index(name) + 1 for name in names]
            columns = [col for col in columns if any(numeric(summary.cell(row, col).value) for row in rows)]
            if not columns:
                continue
            chart = BarChart()
            multiple = sum((s, l) == (stage, light) for s, l, _ in blocks) > 1
            cohort_label = f" · {cohort}" if multiple else ""
            chart.title = f"{light.title()} {stage}{cohort_label} · {title}"
            chart.y_axis.title = unit
            chart.y_axis.scaling.min = 0
            if limit is not None:
                chart.y_axis.scaling.max = limit
            for col in columns:
                reference = Reference(summary, min_col=col, min_row=row_start, max_row=row_end)
                chart.append(SeriesFactory(reference, title=str(summary.cell(1, col).value)))
            chart.set_categories(Reference(summary, min_col=1, min_row=row_start, max_row=row_end))
            chart.width, chart.height = 19, 9
            if len(columns) == 1:
                chart.legend = None
            charts.add_chart(chart, f"{'A' if plotted % 2 == 0 else 'L'}{position + 20 * (plotted // 2)}")
            plotted += 1
        position += 20 * math.ceil(plotted / 2)
        total += plotted
    return total


def export(version, runs, output):
    derived = [record for model, run, sources in runs
               for path, content in sources for record in results_from(model, run, path, content)]
    latest = latest_results(derived)
    book = Workbook()
    guide = book.active
    guide.title = "Guide"
    guide.column_dimensions["A"].width = 105
    for line in (
        f"ANNAlyzer {version} benchmark",
        "Latest results: the newest completed run per model, lighting and evaluation stage.",
        "Charts: editable Excel comparisons. Validation and held-out test stay separate.",
        "All runs: known aggregate metrics and resource peaks, including historical ablations.",
        "Numbers: original finite numeric JSON fields, including timings, training histories, and additional formats.",
        "Artifacts: local links. Images and full-resolution NPY maps remain in their run folders.",
        "AUPRO integrates to 30% background-pixel FPR for current benchmarks.",
        "Regular V2–V4 localization uses the three fixed-pose defects. Red and green pool both classes.",
        "Missing measurements are blank. Compare cohorts and protocols before interpreting differences.",
    ):
        guide.append([line])
    guide["A1"].font = Font(size=18, bold=True, color="164B44")
    summary = sheet(book, "Latest results", HEADERS, map(result_row, latest), output)
    chart_count = add_charts(book, summary, latest)
    sheet(book, "All runs", HEADERS, map(result_row, derived), output)
    sheet(book, "Defect sizes", (*ID, "Region area px²", "Components", "AUPRO", "Overlap at 1% FPR"),
          ((*r.id, label, item.get("components"), item.get("aupro"), item.get("mean_overlap_at_1pct_fpr"))
           for r in derived for key, label in SIZES if (item := r.metrics.get("size_specific", {}).get(key))), output)
    sheet(book, "Per image", (*ID, "Sample", "Cohort", "Score", "Flagged", "Inference seconds", "Map max", "Source JSON"),
          ((*r.id, frame.get("id", frame.get("sample_id")), frame.get("cohort", frame.get("role")),
            frame.get("image_score", frame.get("score")), frame.get("flagged"),
            frame.get("inference_seconds", frame.get("inference_s", frame.get("tile_inference_s"))),
            frame.get("map_max"), r.source) for r in derived for frame in r.images), output)
    sheet(book, "Resources", (*ID, "Measured operation", "Seconds", "RAM MiB", "VRAM allocated MiB", "VRAM reserved MiB"),
          ((*r.id, name, timing.get("seconds"), timing.get("rss_peak_mib", timing.get("rss_peak_mib_sampled")),
            timing.get("cuda_peak_allocated_mib"), timing.get("cuda_peak_reserved_mib"))
           for r in derived for name, timing in r.stages.items() if isinstance(timing, dict)), output)
    sources = [(model, run, path) for model, run, _ in runs for path in json_sources(run)]
    sheet(book, "Training", ("Model", "Run", "JSON field", "Value", "Source JSON"),
          ((model, run.name, field, value, path) for model, run, path in sources
           if path.name in ("training.json", "history.json") for field, value in numeric_fields(read_json(path))), output)
    sheet(book, "Numbers", ("Model", "Run", "JSON field", "Value", "Source JSON"),
          ((model, run.name, field, value, path) for model, run, path in sources
           for field, value in numeric_fields(read_json(path))), output)
    sheet(book, "Artifacts", ("Model", "Run", "Path", "Open file"),
          ((model, run.name, path.relative_to(run).as_posix(), path) for model, run, _ in runs
           for path in sorted(run.rglob("*")) if path.is_file()
           and path.suffix in (".png", ".csv", ".json", ".pt", ".npy", ".zip")
           and not any(part.startswith(("test_pending_", ".")) for part in path.relative_to(run).parts)), output)

    target = output / f"{version}_results.xlsx"
    pending = target.with_suffix(f".{os.getpid()}.pending")
    try:
        book.save(pending)
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)
    print(f"{version}: {len(runs)} saved runs, {len(derived)} metric rows, {chart_count} charts · {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", type=str.upper, help="Limit the export, for example V3 V4")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Anamoly_detection/reports/excel")
    args = parser.parse_args()
    versions = discover(ROOT)
    missing = set(args.versions or ()) - versions.keys()
    if missing:
        print("No completed reports for " + ", ".join(sorted(missing)))
    selected = {version: runs for version, runs in versions.items() if not args.versions or version in args.versions}
    if not selected:
        parser.exit(1, "No completed reports match the requested versions.\n")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for version, runs in selected.items():
        export(version, runs, output)


if __name__ == "__main__":
    main()
