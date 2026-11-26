"""Select lighting conditions for the PatchCore, AnomalyDINO and Flownomaly launchers."""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import v3_workflow as v3


HERE = Path(__file__).resolve().parent
MODELS = {"patchcore": "PatchCore V3", "anomalydino": "AnomalyDINO V3", "flownomaly": "Flownomaly V4"}


def choose_lights(model, available=v3.COLOURS):
    return lighting_checklist(MODELS[model], available)


def lighting_checklist(label, available):
    selected = set(available)
    choices = {str(index): light for index, light in enumerate(v3.COLOURS, 1) if light in available}
    while True:
        print(f"\n{label} lighting conditions")
        for number, light in choices.items():
            print(f"  {number}. [{'x' if light in selected else ' '}] {light.title()}")
        try:
            answer = input(f"Toggle {', '.join(choices)} | A toggles all | Enter confirms | Q cancels: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ()
        if answer == "q":
            return ()
        if not answer:
            if selected:
                return tuple(light for light in v3.COLOURS if light in selected)
            print("Select at least one lighting condition.")
        elif answer == "a":
            selected = set() if len(selected) == len(available) else set(available)
        elif answer in choices:
            selected.symmetric_difference_update((choices[answer],))
        else:
            print("Choose an available number, A or Q.")


def selected_v3(model, lights):
    """Use the V3 fitting helpers for a subset without changing the shared U-Net entry point."""
    dataset, hashes = v3.cohorts()
    folder = HERE / model
    runner = v3.load_module(folder / "V2/run_v2.py", f"{model}_v3_adapter")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = folder / "results/V3" / f"{'_'.join(lights)}_v3_{stamp}"
    output.mkdir(parents=True)
    shutil.copyfile(v3.shared.DATA / "manifest_v2.json", output / "manifest_v2.json")
    shutil.copyfile(v3.shared.SPLIT, output / "split_v1.json")
    print(f"Run directory: {output}", flush=True)
    for light in lights:
        cohort = dataset[light]
        location = output / light
        location.mkdir()
        checkpoint = location / "fitted_state.pt"
        label = v3.MODELS[model]
        print(f"Fitting {label} · {light}", flush=True)
        runtime = runner.build(checkpoint, cohort["train"], cohort["defects"])
        runtime.configuration["protocol"] = f"V3 {light} normal-only {label}"
        v3.save_validation(output, light, cohort, runtime, checkpoint, label, hashes)
        del runtime
    sources = v3.shared.source_hashes(folder)
    for source in (HERE / "v3_workflow.py", Path(__file__).resolve()):
        sources[source.relative_to(v3.shared.ROOT).as_posix()] = v3.shared.sha(source)
    v3.shared.write_json(output / "complete.json", {"model": v3.MODELS[model], "stage": "validation",
                                                     "lights": lights, "dataset_sha256": hashes,
                                                     "source_sha256": sources})
    print(f"Completed V3 selected lighting conditions: {output}", flush=True)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("action", nargs="?", default="validate", choices=("validate", "check", "check-env", "smoke"))
    parser.add_argument("--lighting", nargs="+", choices=("all", *v3.COLOURS), default=["all"])
    options = parser.parse_args(args)
    if options.action == "smoke" and options.model != "flownomaly":
        parser.error("The standalone smoke command is available for Flownomaly.")
    if "all" in options.lighting and len(options.lighting) > 1:
        parser.error("Use all by itself or list individual lighting conditions.")
    lights = (v3.COLOURS if "all" in options.lighting else
              tuple(light for light in v3.COLOURS if light in options.lighting))
    if len(args) == 1:
        lights = choose_lights(options.model)
        if not lights:
            print("Run cancelled.")
            return 0
    if options.model == "flownomaly":
        script, parameters = HERE / "flownomaly/run_v4.py", [options.action, "--lighting", *lights]
    elif options.action != "validate" or lights == v3.COLOURS:
        script, parameters = HERE / "v3_workflow.py", [options.model, options.action]
    else:
        selected_v3(options.model, lights)
        return 0
    return subprocess.run([sys.executable, "-B", "-u", str(script), *parameters], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
