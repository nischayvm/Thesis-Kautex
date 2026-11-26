"""Publish selected completed V2–V5 validation artifacts and the journal."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import update_journal_from_results as journal_importer


ROOT = Path(__file__).resolve().parents[1]
ANOMALY = ROOT / "Anamoly_detection"
EXPORTS = ANOMALY / "benchmark_exports"
MODELS = journal_importer.MODELS
RUN_NAME = journal_importer.V2_RUN
LIGHTS = ("red", "green", "regular")
VERSIONS = ("V2", "V3", "V4", "V5")
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


def available_reports(version):
    for model, label in MODELS.items():
        folder = ANOMALY / model / "results" / version
        for light in LIGHTS:
            if version == "V2":
                options = folder.glob("regular_v2_*/validation/report.json") if light == "regular" else ()
            else:
                options = (file for file in folder.glob(f"*/{light}/validation/report.json")
                           if (file.parents[2] / "complete.json").is_file()
                           and light in json.loads((file.parents[2] / "complete.json").read_text(encoding="utf-8"))["lights"])
            candidates = []
            for file in options:
                path = file.relative_to(ROOT).as_posix()
                if journal_importer.REPORT_PATH.fullmatch(path):
                    run = file.parents[1 if version == "V2" else 2]
                    require(run.resolve().parent == folder.resolve(), f"Unexpected run location: {path}")
                    candidates.append((run.name[-23:], run, file))
            if candidates:
                _, run, path = max(candidates)
                yield model, label, light, run, path


def completed_runs(reports, selected):
    for model, label, light, run, report_path in reports:
        if light not in selected:
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require((report["model"], report["stage"], report.get("lighting", "regular")) == (label, "validation", light),
                f"Unexpected validation report: {report_path.relative_to(ROOT)}")
        checkpoint = (run / report["checkpoint"]["file"]).resolve()
        require(checkpoint.is_relative_to(run.resolve()) and checkpoint.suffix == ".pt"
                and digest(checkpoint) == report["checkpoint"]["sha256"], f"Checkpoint differs from the report: {checkpoint.name}")
        summaries = [report_path.parent / name for name in SUMMARIES]
        heatmaps = sorted((report_path.parent / "visuals").glob("*_heatmap.png"))
        artifacts = [*summaries, checkpoint, *heatmaps]
        diagnostics = [run / "smoke.json", run / light / "smoke.json", run / light / "training.json"]
        artifacts.extend(path for path in diagnostics if path.is_file())
        if model == "flownomaly":
            config = report["configuration"]
            encoder = (run / light / config["encoder_state"]).resolve()
            require(encoder.is_relative_to(run.resolve()) and encoder.suffix == ".pt"
                    and digest(encoder) == config["encoder_state_sha256"], "Flownomaly ViT checkpoint differs from the report")
            artifacts.append(encoder)
        for path in artifacts:
            require(path.is_file() and path.resolve().is_relative_to(run.resolve()),
                    f"Missing or invalid artifact: {path.relative_to(ROOT)}")
        suffix = "" if run.parent.name == "V2" else f"_{light}"
        bundle = EXPORTS / f"{model}_{run.name}{suffix}_validation_artifacts.zip"
        print(f"{label} · {light}: {run.name}, {len(heatmaps)} heatmaps",
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
        rf"Anamoly_detection/(?:benchmark_exports/(?:{models})_(?:{RUN_NAME}|{journal_importer.MODERN_RUN}_(?:red|green|regular))"
        rf"_validation_artifacts\.zip|(?:{models})/results/"
        rf"(?:V2/{RUN_NAME}|V[345]/{journal_importer.MODERN_RUN}/(?:red|green|regular))/validation/(?:{summaries}))", path,
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


def choose_publication():
    sys.path.insert(0, str(ANOMALY))
    from lighting_launcher import lighting_checklist

    runs = {version: list(available_reports(version)) for version in VERSIONS}
    choices = {str(index): version for index, version in enumerate(VERSIONS, 1) if runs[version]}
    require(choices, "No completed validation reports are available to publish.")
    default = next(reversed(choices.values()))
    print("\nSaved validation versions")
    for number, version in choices.items():
        print(f"  {number}. {version}")
    while True:
        try:
            answer = input(f"Choose a version number | Enter uses {default} | Q cancels: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None, ()
        if answer == "q":
            return None, ()
        if not answer or answer in choices:
            version = choices.get(answer, default)
            break
    reports = runs[version]
    lights = tuple(light for light in LIGHTS if any(row[2] == light for row in reports))
    return reports, lighting_checklist(f"Publish {version}", lights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-journal", action="store_true",
                        help="Import the selected validation reports and publish JOURNAL.md with the results")
    parser.add_argument("--dry-run", action="store_true", help="Verify and list results without writing or contacting GitHub")
    parser.add_argument("--interactive", action="store_true", help="Choose a version and lighting interactively")
    parser.add_argument("--version", choices=VERSIONS, default="V2")
    parser.add_argument("--lighting", nargs="+", choices=("all", *LIGHTS), default=["all"])
    args = parser.parse_args()
    if args.interactive:
        reports, lights = choose_publication()
        if not lights:
            print("Publication cancelled.")
            return
    else:
        if "all" in args.lighting and len(args.lighting) > 1:
            parser.error("Use all by itself or list individual lighting conditions.")
        reports = list(available_reports(args.version))
        available = {row[2] for row in reports}
        lights = available if "all" in args.lighting else set(args.lighting)
        require(lights <= available, f"{args.version} has no completed reports for: {', '.join(sorted(lights - available))}")
    runs = list(completed_runs(reports, lights))
    require(runs, "No completed validation reports found for this selection.")
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
        message = "Publish benchmark validation results and journal" if args.update_journal else "Publish benchmark validation results"
        git("commit", "--only", "-m", message, "--", *paths)
    git("lfs", "push", "origin", "main")
    git("push", "origin", "main")
    status = "Validation results and journal" if args.update_journal else "Validation results"
    print(f"{status} are on GitHub.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from None
