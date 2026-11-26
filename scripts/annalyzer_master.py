"""Choose, configure, and execute local Windows ANNAlyzer jobs."""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlretrieve

SCRIPTS = Path(__file__).resolve().parent
ANOMALY = SCRIPTS.parent / "Anamoly_detection"
sys.path.insert(0, str(ANOMALY))
from workstation_config import DEFAULT, LOCAL, MODELS, ROOT, custom_dataset, read_json, rooted, validate, write_json


PYTHON_VERSION = "3.12.13"
PROJECTS = {name: ANOMALY / name / "V1" for name in MODELS}
PROJECTS["flownomaly"] = ANOMALY / "flownomaly"
REVISION = "4610ca143709d58a633b6397a74412c2c3842454"
DINO_CACHE = ANOMALY / "anomalydino/.hf-cache/hub/models--timm--vit_small_patch14_dinov2.lvd142m"
WEIGHTS = {
    "patchcore": (ANOMALY / "patchcore/V1/wide_resnet50_2-9ba9bcbe.pth",
                  "https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
                  "9ba9bcbebc349d733a72eb7608143fd754e4689ac8d8ce2916ca0bdff6443950"),
    "unet": (ANOMALY / "unet/V1/resnet18-f37072fd.pth",
             "https://download.pytorch.org/models/resnet18-f37072fd.pth",
             "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
    "anomalydino": (DINO_CACHE / "snapshots" / REVISION / "model.safetensors",
                    f"https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/resolve/{REVISION}/model.safetensors",
                    "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"),
}


def invoke(*args, env=None, log=None):
    command = list(map(str, args))
    print(subprocess.list2cmdline(command), flush=True)
    if log is None:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"Log {log}", flush=True)
        with (log.open("x", encoding="utf-8") as file,
              subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process):
            for line in process.stdout:
                print(line, end="", flush=True)
                file.write(line)
                file.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)


def env_python(name):
    env = LOCAL / "envs" / name / "Scripts/python.exe"
    if not env.is_file():
        raise FileNotFoundError(f"{env} is missing. Run the master setup on this computer.")
    return env


def gui_python():
    candidates = (LOCAL / "envs/inspection/Scripts/python.exe",
                  ANOMALY / "inspection_app/.venv/Scripts/python.exe", Path(sys.executable))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            result = subprocess.run([str(candidate), "-B", "-c", "from PyQt6 import QtWidgets"],
                                    capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return candidate
    return None


def launch_gui(path):
    python = gui_python()
    if python is None:
        if input("Install the isolated PyQt6 environment to open the workstation window? [Y/n] ").strip().lower() not in ("", "y", "yes"):
            return
        setup(["inspection"])
        python = env_python("inspection")
    log = log_file("ui_startup")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as output:
        subprocess.Popen([str(python), "-B", str(SCRIPTS / "annalyzer_workstation_ui.py"), "--config", str(path)],
                         cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def setup(components):
    uv = shutil.which("uv")
    tools = LOCAL / "bootstrap"
    if uv is None:
        uv = tools / "Scripts/uv.exe"
        if not uv.is_file():
            invoke(sys.executable, "-m", "venv", tools)
            invoke(tools / "Scripts/python.exe", "-m", "pip", "install", "uv==0.11.19")
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(LOCAL / "python"), UV_CACHE_DIR=str(LOCAL / "cache"),
               UV_NO_PROGRESS="1", UV_LINK_MODE="copy")
    invoke(uv, "python", "install", PYTHON_VERSION, env=env)
    for component in components:
        destination = LOCAL / "envs" / component
        if component in PROJECTS:
            invoke(uv, "sync", "--locked", "--project", PROJECTS[component], "--python", PYTHON_VERSION,
                   "--managed-python", env=dict(env, UV_PROJECT_ENVIRONMENT=str(destination)))
        else:
            requirements = (SCRIPTS / "workstation-requirements.txt" if component == "reporting"
                            else ANOMALY / "inspection_app/requirements.txt")
            if not (destination / "Scripts/python.exe").is_file():
                invoke(uv, "venv", destination, "--python", PYTHON_VERSION, "--managed-python", env=env)
            invoke(uv, "pip", "install", "--python", env_python(component), "-r", requirements, env=env)
        invoke(env_python(component), "--version")


def download_weights(models):
    def digest(path):
        checksum = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                checksum.update(block)
        return checksum.hexdigest()

    names = set(models) - {"flownomaly"}
    if "flownomaly" in models:
        names.add("anomalydino")
    for name in WEIGHTS:
        if name not in names:
            continue
        path, url, expected = WEIGHTS[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or (path.stat().st_size < 1024 and b"git-lfs.github.com/spec" in path.read_bytes()):
            pending = path.with_suffix(".pending")
            try:
                print(f"Downloading {name} pretrained weights", flush=True)
                urlretrieve(url, pending)
                if digest(pending) != expected:
                    raise ValueError(f"Unexpected download checksum for {name}")
                pending.replace(path)
            finally:
                pending.unlink(missing_ok=True)
        if digest(path) != expected:
            raise ValueError(f"Existing weights differ from the pinned checksum: {path}")
        if name == "anomalydino":
            refs = DINO_CACHE / "refs"
            refs.mkdir(exist_ok=True)
            (refs / "main").write_text(REVISION, encoding="ascii")
        print(f"Verified {path}", flush=True)


def blender(config):
    if config["blender"]:
        exe = rooted(config["blender"])
        if exe.is_file():
            return exe
        raise FileNotFoundError(exe)
    detected = shutil.which("blender.exe")
    if detected:
        return detected
    standard = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Blender Foundation"
    for candidate in [*sorted(standard.glob("*/blender.exe"), reverse=True),
                      Path("E:/SteamLibrary/steamapps/common/Blender/blender.exe")]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Blender was not found. Set the blender field to the blender.exe path.")


def render(config, path, resume=False, demo=None):
    scene = rooted(config["scene"])
    if not scene.is_file():
        raise FileNotFoundError(scene)
    if demo:
        light, kind = demo
        runner = SCRIPTS / "render_annalyzer_demo.py"
        settings = config["render"]
        args = ["--lighting", light, "--kind", kind, "--seed", settings["seed"],
                "--resolution", settings["resolution"], "--samples", settings["cycles_samples"],
                "--gpu-match", settings["gpu_match"]]
        for height in ("camera_height", "led_height"):
            if settings[height] is not None:
                args.extend(("--" + height.replace("_", "-"), settings[height]))
    else:
        runner = SCRIPTS / "render_annalyzer_batch.py"
        args = ["--workstation-config", path, *(("--resume",) if resume else ())]
    invoke(blender(config), "--background", scene, "--python-exit-code", "1", "--python", runner, "--",
           *args, log=log_file("render"))


def log_file(label):
    return LOCAL / "logs" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{label}.log"


def edit(path):
    config = deepcopy(validate(read_json(path)))
    while True:
        flatten = {}
        for key, value in config.items():
            if isinstance(value, dict):
                flatten.update((f"{key}.{item}", (value, item)) for item in value)
            else:
                flatten[key] = (config, key)
        print("\nEditable settings")
        for key, (parent, field) in flatten.items():
            print(f"  {key} = {json.dumps(parent[field], ensure_ascii=False)}")
        option = input("Type a setting name, S to save, or Q to discard changes: ").strip()
        if option.casefold() == "q":
            return
        if option.casefold() == "s":
            try:
                validate(config)
            except ValueError as exc:
                print(exc)
                continue
            write_json(path, config)
            return
        if option not in flatten:
            print("Select a setting from the list.")
            continue
        parent, field = flatten[option]
        entered = input("New value (JSON for lists, numbers, true/false or null): ").strip()
        current = parent[field]
        if isinstance(current, str):
            parent[field] = entered.strip('"')
        else:
            try:
                parent[field] = json.loads(entered)
            except ValueError:
                print("Enter valid JSON, for example [\"red\", \"green\"].")


def create(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(DEFAULT, stream, indent=2)
        stream.write("\n")
    print(f"Saved editable job {path}", flush=True)


def model_job(model, action, path=None):
    args = (path,) if path is not None else ()
    invoke(env_python(model), "-B", "-u", ANOMALY / "workstation_experiment.py", model, action, *args,
           log=log_file(f"{model}_{action}"))


def dispatch(action, path, args):
    if action == "new":
        return create(path)
    if action in ("validate", "test"):
        if not args.model or not args.run:
            raise ValueError("Specify --model and --run for saved-checkpoint evaluation.")
        return model_job(args.model, action, rooted(args.run))
    config = validate(read_json(path))
    models = config["training"]["models"]
    if action == "setup":
        setup(args.components or [*models, "reporting", "inspection"])
    elif action == "weights":
        download_weights(models)
    elif action == "check":
        if shutil.which("nvidia-smi"):
            invoke("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv")
        for model in models:
            model_job(model, "check")
    elif action in ("render", "resume"):
        render(config, path, resume=action == "resume")
    elif action == "demo":
        if args.light is None or args.kind is None:
            raise ValueError("Provide the demo lighting and choose clean or blistered.")
        render(config, path, demo=(args.light, args.kind))
    elif action == "audit":
        invoke(env_python("reporting"), "-B", "-u", __file__, "audit-local", "--config", path,
               log=log_file("audit"))
    elif action == "audit-local":
        from workstation_dataset import audit

        audit(config, path.with_suffix(".split.json"))
    elif action == "fit":
        for model in models:
            model_job(model, "fit", path)
    elif action == "inspect":
        python = gui_python()
        if python is None:
            raise ValueError("Install the inspection environment before opening tray inspection.")
        app_env = dict(os.environ, ANNALYZER_MODEL_ENV_ROOT=str(LOCAL / "envs"))
        invoke(python, "-B", ANOMALY / "inspection_app/app.py", env=app_env)
    elif action in ("html","excel","journal","publish"):
        scripts = {"html": "generate_benchmark_html_report.py", "excel": "generate_benchmark_excel.py",
                   "journal": "update_journal_from_results.py", "publish": "publish_v2_results.py"}
        extra = ["--interactive", "--update-journal"] if action == "publish" else []
        invoke(env_python("reporting"), "-B", "-u", SCRIPTS / scripts[action], *extra)
    elif action == "benchmark":
        for model in models:
            if model == "flownomaly":
                runner = ANOMALY / "flownomaly/run_v4.py"
                options = ["validate", "--lighting", *config["training"]["lights"]]
            elif model == "unet":
                runner = ANOMALY / "v3_workflow.py"
                options = [model, "validate"]
            else:
                runner = ANOMALY / "lighting_launcher.py"
                options = [model, "validate", "--lighting", *config["training"]["lights"]]
            invoke(env_python(model), "-B", "-u", runner, *options, log=log_file(f"{model}_benchmark"))


MENU = (
    ("edit", "Edit job settings"), ("open", "Open or create a different job file"),
    ("setup", "Install requirements and create isolated Python environments"),
    ("weights", "Download or verify pretrained weights"), ("check", "Check workstation GPU and model environments"),
    ("demo", "Render one demo image and mask"), ("render", "Render a custom paired dataset"),
    ("resume", "Resume an interrupted dataset render"), ("audit", "Audit the dataset and save grouped splits"),
    ("fit", "Fit selected models and validate"), ("benchmark", "Run original V3/V4 benchmark"),
    ("validate", "Re-run validation from a saved custom checkpoint"),
    ("test", "Evaluate a completed custom run on its held-out fold"),
    ("html", "Generate HTML report"), ("excel", "Generate Excel reports"), ("inspect", "Open tray inspection"),
    ("journal", "Update the results journal"), ("publish", "Publish selected saved results to GitHub"),
)


def menu(path, args):
    if not path.exists():
        create(path)
    while True:
        print(f"\nANNAlyzer workstation\nJob {path}")
        for index, (_, label) in enumerate(MENU, 1):
            print(f" {index:2}. {label}")
        choice = input("Select an option, or Q to quit: ").strip()
        if choice.casefold() == "q":
            return
        if not choice.isdigit() or not 1 <= int(choice) <= len(MENU):
            continue
        action = MENU[int(choice) - 1][0]
        try:
            if action == "edit":
                edit(path)
                continue
            if action == "open":
                entry = input("Path to the job JSON: ").strip().strip('"')
                if entry:
                    selected = rooted(entry)
                    if not selected.exists():
                        create(selected)
                    path = selected
                continue
            if action == "demo":
                args.light = input("Lighting (red, green, regular): ").strip().lower()
                args.kind = input("Kind (clean, blistered): ").strip().lower()
            if action in ("validate", "test"):
                args.model = input(f"Model ({', '.join(MODELS)}): ").strip().lower()
                if args.model not in MODELS:
                    raise ValueError("Choose a listed model.")
                args.run = input("Completed V5 run directory: ").strip().strip('"')
            if action in ("setup", "weights", "render", "resume", "fit", "benchmark", "test", "publish"):
                if input(f"Start {action}? Type YES to continue: ").strip().upper() != "YES":
                    continue
            dispatch(action, path, args)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"Job stopped. {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = tuple(action for action, _ in MENU if action not in ("edit", "open"))
    parser.add_argument("action", nargs="?", default="gui", choices=("gui", "menu", "new", "audit-local", *commands))
    parser.add_argument("--config", type=Path, default=LOCAL / "jobs/default.json")
    parser.add_argument("--components", nargs="+", choices=(*MODELS,"reporting","inspection"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--run")
    parser.add_argument("--light", choices=("red","green","regular"))
    parser.add_argument("--kind", choices=("clean","blistered"))
    args = parser.parse_args()
    path = rooted(args.config)
    if args.action == "gui":
        launch_gui(path)
    elif args.action == "menu":
        menu(path, args)
    else:
        if not path.exists() and args.action in ("setup","check","weights","html","excel","inspect","journal","publish"):
            create(path)
        dispatch(args.action, path, args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, EOFError, RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Job stopped. {exc}")
