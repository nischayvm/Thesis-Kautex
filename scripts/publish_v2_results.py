"""Publish completed V2 validation reports and artifact bundles to GitHub."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import update_journal_from_results as journal_importer


ROOT = Path(__file__).resolve().parents[1]
ANOMALY = ROOT / "Anamoly_detection"
EXPORTS = ANOMALY / "benchmark_exports"
MODELS = {"patchcore": "PatchCore", "anomalydino": "AnomalyDINO", "unet": "U-Net"}
RUN_NAME = r"regular_v2_\d{8}T\d{6}_\d{6}Z"
SUMMARIES = ("report.json", "scores.csv", "image_scores.png")
JOURNAL_PATH = "Anamoly_detection/JOURNAL.md"
REMOTE_URLS = {
    "https://github.com/nischayvm/Thesis-Kautex.git",
    "https://github.com/nischayvm/Thesis-Kautex",
    "git@github.com:nischayvm/Thesis-Kautex.git",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(*args, capture=False, allowed=(0,)):
    command = ["git", *args]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE if capture else None)
    if result.returncode not in allowed:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.stdout.strip() if capture else result.returncode


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def completed_runs():
    for model, label in MODELS.items():
        folder = ANOMALY / model / "results/V2"
        runs = sorted(folder.glob("regular_v2_*/validation/report.json"), reverse=True)
        if not runs:
            print(f"{label}: no completed V2 validation run", flush=True)
            continue
        report_path = runs[0]
        run = report_path.parents[1]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(re.fullmatch(RUN_NAME, run.name) and run.resolve().parent == folder.resolve()
                and report["model"] == label and report["stage"] == "validation",
                f"Unexpected validation report: {report_path.relative_to(ROOT)}")
        checkpoint = run / "regular_state.pt"
        require(report["checkpoint"]["file"] == checkpoint.name
                and digest(checkpoint) == report["checkpoint"]["sha256"],
                f"Checkpoint differs from validation report: {checkpoint.relative_to(ROOT)}")
        summaries = [report_path.parent / name for name in SUMMARIES]
        heatmaps = sorted((report_path.parent / "visuals").glob("*_heatmap.png"))
        artifacts = [*summaries, run / "smoke.json", checkpoint, *heatmaps]
        for path in artifacts:
            require(path.is_file() and path.resolve().is_relative_to(run.resolve()),
                    f"Missing or invalid artifact: {path.relative_to(ROOT)}")
        bundle = EXPORTS / f"{model}_{run.name}_validation_artifacts.zip"
        print(f"{label}: {run.name}, {len(heatmaps)} heatmaps, {checkpoint.stat().st_size / 1024**2:.1f} MiB checkpoint",
              flush=True)
        yield run, summaries, artifacts, bundle


def write_bundle(run, files, bundle):
    expected = {path.relative_to(run).as_posix(): path for path in files}
    if bundle.exists():
        with ZipFile(bundle) as archive:
            require(set(archive.namelist()) == set(expected), f"Existing bundle has different files: {bundle.name}")
            for name, path in expected.items():
                with archive.open(name) as source:
                    require(hashlib.file_digest(source, "sha256").hexdigest() == digest(path),
                            f"Existing bundle differs from {path.name}: {bundle.name}")
        return
    with NamedTemporaryFile(dir=bundle.parent, suffix=".pending", delete=False) as file:
        pending = Path(file.name)
    try:
        with ZipFile(pending, mode="w") as archive:
            for name, path in expected.items():
                archive.write(path, name)
        pending.replace(bundle)
    finally:
        pending.unlink(missing_ok=True)


def is_publish_path(path):
    models = "|".join(MODELS)
    summaries = "|".join(re.escape(name) for name in SUMMARIES)
    return re.fullmatch(
        rf"Anamoly_detection/(?:benchmark_exports/(?:{models})_{RUN_NAME}_validation_artifacts\.zip"
        rf"|(?:{models})/results/V2/{RUN_NAME}/validation/(?:{summaries}))", path,
    ) is not None


def check_branch(include_journal):
    require(git("branch", "--show-current", capture=True) == "main", "Switch to main before publishing results.")
    remotes = git("remote", "get-url", "--push", "--all", "origin", capture=True).splitlines()
    require(len(remotes) == 1 and remotes[0] in REMOTE_URLS, "origin must point to nischayvm/Thesis-Kautex.")
    git("fetch", "origin", "main")
    behind, _ = map(int, git("rev-list", "--left-right", "--count", "origin/main...HEAD", capture=True).split())
    require(behind == 0, "Local main is behind origin/main. Sync the branches before publishing results.")
    outgoing = git("log", "--format=", "--name-only", "--no-renames", "origin/main..HEAD", capture=True).splitlines()
    require(all(is_publish_path(path) or (include_journal and path == JOURNAL_PATH) for path in outgoing if path),
            "Local main has unrelated outgoing changes. Push those separately before publishing results.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-journal", action="store_true",
                        help="Import the selected validation reports and publish JOURNAL.md with the results")
    parser.add_argument("--dry-run", action="store_true", help="Verify and list results without writing or contacting GitHub")
    args = parser.parse_args()
    runs = list(completed_runs())
    require(runs, "No completed V2 validation reports found. Run a model launcher first.")
    if args.update_journal:
        selected = {summaries[0].relative_to(ROOT).as_posix() for _, summaries, _, _ in runs}
        journal_source, journal_reports = journal_importer.reports(False, selected)
    if args.dry_run:
        if args.update_journal:
            _, count = journal_importer.journal_section(
                journal_source, journal_reports, (ROOT / JOURNAL_PATH).read_text(encoding="utf-8"),
            )
            print(f"Journal: {count} new summaries. The push also includes any local journal edits.", flush=True)
        print("Dry run complete. Git and the artifact folders were untouched.", flush=True)
        return

    check_branch(args.update_journal)
    git("lfs", "version")
    paths, bundles = [], []
    for run, summaries, artifacts, bundle in runs:
        relative_bundle = bundle.relative_to(ROOT).as_posix()
        attributes = git("check-attr", "filter", "--", relative_bundle, capture=True)
        require(attributes.endswith(": filter: lfs"), f"The existing Git LFS rule does not cover {relative_bundle}")
        write_bundle(run, artifacts, bundle)
        paths.extend(path.relative_to(ROOT).as_posix() for path in (*summaries, bundle))
        bundles.append(relative_bundle)

    if args.update_journal:
        journal_importer.append_to_journal(journal_source, journal_reports)
        paths.append(JOURNAL_PATH)
    git("add", "--force", "--", *paths)
    for path in bundles:
        size = int(git("cat-file", "-s", f":{path}", capture=True))
        require(size < 1024 and git("show", f":{path}", capture=True).startswith("version https://git-lfs.github.com/spec/v1\n"),
                f"Git did not stage an LFS pointer for {path}. Check the Git LFS installation before publishing.")
    if git("diff", "--cached", "--quiet", "--", *paths, allowed=(0, 1)) == 1:
        message = "Publish V2 validation results and journal" if args.update_journal else "Publish V2 validation results"
        git("commit", "--only", "-m", message, "--", *paths)
    git("lfs", "push", "origin", "main")
    git("push", "origin", "main")
    status = "V2 validation results and journal" if args.update_journal else "V2 validation results"
    print(f"{status} are on GitHub.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from None
