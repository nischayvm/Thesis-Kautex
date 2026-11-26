"""Append completed benchmark result summaries to the project journal."""

import argparse
import json
import os
import re
import subprocess
from datetime import date
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "Anamoly_detection" / "JOURNAL.md"
MODELS = {"patchcore": "PatchCore", "anomalydino": "AnomalyDINO", "unet": "U-Net", "flownomaly": "Flownomaly"}
TIME_STAMP = r"\d{8}T\d{6}_\d{6}Z"
V2_RUN = rf"regular_v2_{TIME_STAMP}"
MODERN_RUN = rf"[a-z_]+_v[34]_{TIME_STAMP}"
REPORT_PATH = re.compile(
    rf"Anamoly_detection/(?P<model>{'|'.join(MODELS)})/results/"
    rf"(?:(?P<v2>V2)/(?P<legacy>{V2_RUN})|(?P<modern>V[34])/(?P<run>{MODERN_RUN})/(?P<light>red|green|regular))/"
    r"(?P<stage>validation|test)/report\.json"
)
GITHUB = "https://github.com/nischayvm/Thesis-Kautex"
SIZES = ("small_lt64", "medium_64_to_511", "large_ge512")


def git(*args):
    return subprocess.check_output(("git", *args), cwd=ROOT)


def reports(from_git, selected_paths=None):
    if from_git:
        git("fetch", "origin", "main")
        revision = git("rev-parse", "origin/main").decode("ascii").strip()
        paths = set(git("ls-tree", "-r", "--name-only", revision, "--", "Anamoly_detection").decode("utf-8").splitlines())
        read = lambda path: git("show", f"{revision}:{path}")
        exists = paths.__contains__
        link = lambda path: f"{GITHUB}/blob/{revision}/{path}"
        source = f"[GitHub main snapshot]({GITHUB}/commit/{revision})"
    else:
        paths = set()
        for model in MODELS:
            results = JOURNAL.parent / model / "results"
            paths.update(file.relative_to(ROOT).as_posix() for file in results.glob("V2/*/*/report.json"))
            paths.update(file.relative_to(ROOT).as_posix() for file in results.glob("V[34]/*/*/*/report.json")
                         if (file.parents[2] / "complete.json").is_file())
        read = lambda path: (ROOT / path).read_bytes()
        exists = lambda path: (ROOT / path).is_file()
        link = lambda path: path.removeprefix("Anamoly_detection/")
        source = "local saved run folders"
    if selected_paths is not None:
        missing = selected_paths - paths
        if missing:
            raise ValueError(f"Reports not found: {', '.join(sorted(missing))}")
        paths = selected_paths
    entries = []
    for path in sorted(paths):
        match = REPORT_PATH.fullmatch(path)
        if not match:
            continue
        model, stage = match.group("model", "stage")
        run = match["run"] or match["legacy"]
        light = match["light"] or "regular"
        try:
            result = json.loads(read(path))
            if (result["model"], result["stage"], result.get("lighting", "regular")) != (MODELS[model], stage, light):
                raise ValueError("Report model, stage or lighting differs from its path")
            artifacts = [("Report", link(path))]
            parent = path.rsplit("/", 1)[0]
            for name, filename in (("Scores CSV", "scores.csv"), ("Score plot", "image_scores.png")):
                artifact = f"{parent}/{filename}"
                if exists(artifact):
                    artifacts.append((name, link(artifact)))
            entries.append((path, run, result, artifacts))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
    return source, entries


def numeric(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def marker(path):
    prefix = "v2-result-report" if "/results/V2/" in path else "benchmark-result-report"
    return f"<!-- {prefix}: {path} -->"


def journal_section(source, entries, previous):
    new = [entry for entry in entries if marker(entry[0]) not in previous]
    if not new:
        return "", 0
    versions = ", ".join(sorted({entry[0].split("/")[3] for entry in new}))
    lines = [f"## Saved {versions} run results · {date.today().isoformat()}", "",
             f"Imported from {source}. The regular pixel metrics pool fixed-pose defective frames. "
             "Red and green pixel metrics pool both clean and defective frames. "
             "Size columns show AUPRO / mean component overlap at 1% background pixel FPR. "
             "AUPRO integrates up to 30% background pixel FPR. Image decisions use the saved threshold "
             "for varied normals and fixed-pose defects.", ""]
    for stage, title in (("validation", "Validation and calibration"), ("test", "Held-out test")):
        group = [entry for entry in new if entry[2]["stage"] == stage]
        if not group:
            continue
        lines.extend([f"### {title}", ""])
        if stage == "validation":
            lines.extend(["The highest normal validation score sets the threshold. "
                          "The validation false-positive rate is zero by construction.", ""])
        lines.extend([
            "| Model / lighting / UTC run ID | Pixel AUROC | Pixel AP | AUPRO | Image AUROC / AP | "
            "Small | Medium | Large |",
            "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
        ])
        for _, run, r, _ in group:
            d = r["defect_cohort"] if "defect_cohort" in r else r["fixed_pose_defects"]
            size = [d["size_specific"][key] for key in SIZES]
            cells = [f'{r["model"]} / {r.get("lighting", "regular").title()} / `{run}`', numeric(d["pixel_auroc"]), numeric(d["pixel_ap"]),
                     numeric(d["aupro"]), f'{numeric(r["image_auroc"])} / {numeric(r["image_ap"])}',
                     *(f'{numeric(s["aupro"])} / {numeric(s["mean_overlap_at_1pct_fpr"])}' for s in size)]
            lines.append("| " + " | ".join(cells) + " |")
        lines.extend([
            "",
            "| Model / lighting / UTC run ID | Threshold | TN / FP | TP / FN | Sensitivity / FPR | "
            "Mean inference (s/frame) | Bank rows × features |",
            "| --- | ---: | ---: | ---: | --- | ---: | --- |",
        ])
        for _, run, r, _ in group:
            n, d = ((r["normal_cohort"], r["defect_cohort"]) if "normal_cohort" in r
                    else (r["varied_normal"], r["fixed_pose_defects"]))
            fp, tp = n["false_positives"], d["true_positives"]
            bank = r.get("configuration", {}).get("bank_dimensions")
            inference = mean(row["inference_seconds"] for row in r["rows"])
            cells = [f'{r["model"]} / {r.get("lighting", "regular").title()} / `{run}`', numeric(r["threshold"], 8), f'{n["frames"] - fp} / {fp}',
                     f'{tp} / {d["frames"] - tp}', f'{d["sensitivity"]:.1%} / {n["false_positive_rate"]:.1%}',
                     numeric(inference, 3), " × ".join(map(str, bank)) if bank else "N/A"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        for path, run, r, artifacts in group:
            refs = " · ".join(f"[{title}]({href})" for title, href in artifacts)
            lines.append(f'- **{r["model"]} / {r.get("lighting", "regular").title()} / `{run}`**. {refs} {marker(path)}')
        lines.append("")
    return "\n".join(lines), len(new)


def append_to_journal(source, entries):
    with JOURNAL.open("r+b") as file:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        original = file.read()
        section, count = journal_section(source, entries, original.decode("utf-8"))
        if count:
            newline = b"\r\n" if b"\r\n" in original else b"\n"
            end = original.rstrip(b"\r\n")
            prefix = newline * max(0, 2 - original[len(end):].count(b"\n"))
            file.seek(0, os.SEEK_END)
            file.write(prefix + section.encode("utf-8").replace(b"\n", newline))
            file.flush()
    if count:
        print(f"Appended {count} completed report summaries to {JOURNAL.relative_to(ROOT)}.")
    else:
        print("No new completed reports to import.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-git", action="store_true",
                        help="Fetch origin/main and read published reports without merging or checking out files")
    parser.add_argument("--dry-run", action="store_true", help="Preview the journal entry without writing it")
    args = parser.parse_args()
    source, entries = reports(args.from_git)
    if args.dry_run:
        section, count = journal_section(source, entries, JOURNAL.read_text(encoding="utf-8"))
        print(section if count else "No new completed reports to import.")
    else:
        append_to_journal(source, entries)


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from None
