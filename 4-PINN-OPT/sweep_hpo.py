"""Joint hyperparameter optimization of every method of 4-PINN-opt, run without arguments by
run_hpo.sh (the flags serve smoke tests): an Optuna TPE search per method, every trial trained
on SEARCH_SEEDS and scored by its mean learning area; the best trial is written into train.py.
adam_aura is searched after adam, on top of its selected betas; muon_aura searches the Muon momentum
together with its gates."""

import argparse
import ast
import dataclasses
import datetime
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src import metrics, optimizer_config  # noqa: E402

TRIALS = 50
# Weight-initialization seeds of every trial; the boundary points are those of train.py.
SEARCH_SEEDS = (100, 101, 102)
STARTUP_TRIALS = 10
SAMPLER_SEED = 0
# Concurrent runs: each worker holds a JAX runtime and gets 0.9/WORKERS of the GPU memory.
WORKERS = 2
MAX_FAILURES = 5
SIGNIFICANT_DIGITS = 4
# Trials whose loss curves the report draws, besides the starting point.
SHOWN_TRIALS = 3
CURVE_WINDOW = 20
# Search value of a trial that went non-finite: at least the learning area of a loss held at 1
# (the squared traction scale).
DIVERGED_LEARNING_AREA = metrics.LOG_LEARNING_AREA_REFERENCE
SWEEP_DIRNAME = "hpo"
LR_MARKERS = ("# BEGIN LR_SWEEP", "# END LR_SWEEP")
METRIC_KEYS = ("learning_area", "final_train_mse", "final_test_mse", "best_test_mse", "training_seconds")


@dataclass(frozen=True)
class Dimension:
    name: str
    low: float
    high: float
    log: bool = True


# Parameters map onto config fields: one_minus_<f> = 1 - f, <f>_minus_one = f - 1,
# chi_gap = chi_alignment - chi_opposition, psi_ratio = psi_alignment / psi_opposition.
LEARNING_RATE = Dimension("learning_rate", 1e-4, 1e-1)
ADAM_BETAS = (Dimension("one_minus_beta_1", 0.001, 0.3), Dimension("one_minus_beta_2", 1e-4, 0.1))
MUON_BETA = Dimension("one_minus_beta", 0.001, 0.3)
# The box of 10-cifar-10-simplified, widened where muon_aura selected a bound (beta_zeta, psi_opposition);
# the gap and the ratio keep scale_by_aura's orderings.
GATES = (Dimension("one_minus_beta_zeta", 0.01, 0.6), Dimension("chi_alignment", 0.5, 0.98, log=False),
         Dimension("chi_gap", 0.02, 1.2), Dimension("psi_opposition", 0.01, 0.9),
         Dimension("psi_ratio", 0.005, 0.9))
SPACES = {
    optimizer_config.METHOD_RPROP: (LEARNING_RATE, Dimension("one_minus_eta_minus", 0.01, 0.6),
                                    Dimension("eta_plus_minus_one", 0.005, 0.5)),
    optimizer_config.METHOD_ADAM: (LEARNING_RATE, *ADAM_BETAS),
    optimizer_config.METHOD_ADAM_VARIABLE_LR: (LEARNING_RATE, *ADAM_BETAS),
    optimizer_config.METHOD_NADAMW: (LEARNING_RATE, *ADAM_BETAS, Dimension("weight_decay", 1e-5, 0.1)),
    optimizer_config.METHOD_CvAMSGrad: (LEARNING_RATE, *ADAM_BETAS),
    optimizer_config.METHOD_MUON: (LEARNING_RATE, MUON_BETA),
    optimizer_config.METHOD_ADAM_AURA: (LEARNING_RATE, *GATES),
    # Not on top of muon: at muon's selected beta (0.997) the default gates diverge.
    optimizer_config.METHOD_MUON_AURA: (LEARNING_RATE, MUON_BETA, *GATES),
}
# Searched once their base method is complete, with its selected tuned fields fixed in every trial.
BASES = {optimizer_config.METHOD_ADAM_AURA: optimizer_config.METHOD_ADAM}


def _significant(value):
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def _field_name(parameter):
    return {"chi_gap": "chi_opposition", "psi_ratio": "psi_alignment"}.get(
        parameter, parameter.removeprefix("one_minus_").removesuffix("_minus_one"))


def values_from_params(params):
    """(learning rate, config fields) of a trial's parameters, rounded to SIGNIFICANT_DIGITS."""
    p = {name: _significant(value) for name, value in params.items()}
    fields = {}
    for name, value in p.items():
        if name.startswith("one_minus_"):
            fields[_field_name(name)] = round(1.0 - value, 12)
        elif name.endswith("_minus_one"):
            fields[_field_name(name)] = round(1.0 + value, 12)
        elif name not in ("learning_rate", "chi_gap", "psi_ratio"):
            fields[name] = value
    if "chi_gap" in p:
        fields["chi_opposition"] = _significant(fields["chi_alignment"] - p["chi_gap"])
    if "psi_ratio" in p:
        fields["psi_alignment"] = _significant(p["psi_ratio"] * fields["psi_opposition"])
    return p["learning_rate"], fields


def params_from_values(method, learning_rate, fields):
    params = {}
    for dimension in SPACES[method]:
        name = dimension.name
        if name == "learning_rate":
            params[name] = learning_rate
        elif name.startswith("one_minus_"):
            params[name] = 1.0 - fields[_field_name(name)]
        elif name.endswith("_minus_one"):
            params[name] = fields[_field_name(name)] - 1.0
        elif name == "chi_gap":
            params[name] = fields["chi_alignment"] - fields["chi_opposition"]
        elif name == "psi_ratio":
            params[name] = fields["psi_alignment"] / fields["psi_opposition"]
        else:
            params[name] = fields[name]
    return params


def _config(train, method):
    return getattr(train, train.CONFIG_CONSTANTS[method][1])


def _field_names(train, method):
    """The tuned config fields of ``method``, in the config's order."""
    tuned = {_field_name(d.name) for d in SPACES[method] if d.name != "learning_rate"}
    return [f.name for f in dataclasses.fields(_config(train, method)) if f.name in tuned]


def _default_values(train, method):
    """The fixed starting point: train.LEARNING_RATE and the field defaults of src/optimizer_config.py."""
    defaults = type(_config(train, method))()
    return (float(train.LEARNING_RATE),
            {field: float(getattr(defaults, field)) for field in _field_names(train, method)})


def _unit(dimension, value):
    if dimension.log:
        return math.log(value / dimension.low) / math.log(dimension.high / dimension.low)
    return (value - dimension.low) / (dimension.high - dimension.low)


def _json_safe(value):
    """``value`` with every non-finite float replaced by None (strict JSON)."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _finite_or_none(values):
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values, dtype=float)]


def _from_none(values):
    return np.asarray([np.nan if v is None else v for v in values], dtype=float)


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.part")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=1, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def _protocol(train, epochs):
    """train.py's training protocol, which a stored run must match to be reused."""
    return {
        "epochs": int(epochs),
        "train_points": int(train.N_TRAIN),
        "test_points": int(train.N_TEST),
        "boundary_seed": int(train.BOUNDARY_SAMPLING_SEEDS[0]),
        "model": {"layer_sizes": [int(n) for n in train.model.LAYER_SIZES], "beta": float(train.model.BETA),
                  "gauss": int(train.model.GAUSS)},
        "geometry": {"half_l": float(train.dataset.HALF_L), "hole_r": float(train.dataset.HOLE_R),
                     "tension": float(train.dataset.TENSION)},
        "adam_variable_lr_schedule": {"epochs": [int(e) for e in train.lr_schedule.SCHEDULER_APPLY_EPOCHS],
                                      "gamma": float(train.lr_schedule.SCHEDULER_GAMMA)},
    }


def _run_config(train, method, learning_rate, fields, seed, epochs):
    keyword = train.CONFIG_CONSTANTS[method][0]
    config = replace(_config(train, method), **fields)
    return {**_protocol(train, epochs), "method": method, "learning_rate": float(learning_rate), "seed": int(seed),
            "hyperparameters": optimizer_config.hyperparameters_for_method(method, **{keyword: config})}


def _run_path(sweep_dir, method, learning_rate, fields, seed):
    digest = hashlib.sha1(json.dumps([learning_rate, fields], sort_keys=True).encode()).hexdigest()[:10]
    return sweep_dir / "runs" / f"{method}_lr{learning_rate:g}_{digest}_seed{seed}.json"


def _load_run(path, expected_config):
    """The stored run at ``path`` if it finished under ``expected_config``, else None."""
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if payload.get("status") != "ok":
        return None
    # Compare in JSON form: stored tuples come back as lists.
    if payload.get("config") != json.loads(json.dumps(expected_config)):
        return None
    return payload


def _worker(method, learning_rate, fields, seed, epochs, out_path):
    import train  # before jax: train.py selects the backend at import

    import jax

    name = train.CONFIG_CONSTANTS[method][1]
    setattr(train, name, replace(getattr(train, name), **fields))
    started = datetime.datetime.now().isoformat(timespec="seconds")
    epochs = epochs or train.N_EPOCHS
    config = _run_config(train, method, learning_rate, fields, seed, epochs)
    try:
        run = train.run_one(method, seed, train.BOUNDARY_SAMPLING_SEEDS[0], (), learning_rate=learning_rate,
                            epochs=epochs, track_dynamics=False)
    except BaseException:
        _write_json_atomic(out_path, {"status": "failed", "config": config, "started": started,
                                      "traceback": traceback.format_exc()})
        raise

    train_losses = np.asarray(run["train_losses"], dtype=float)
    test_losses = np.asarray(run["test_losses"], dtype=float)
    _write_json_atomic(out_path, {
        "status": "ok",
        "config": config,
        "started": started,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "device": jax.devices()[0].device_kind,
        "metrics": {
            "learning_area": float(run["learning_area"]),
            "final_train_mse": float(run["final_train_mse"]),
            "final_test_mse": float(run["final_test_mse"]),
            "best_test_mse": float(run["best_test_mse"]),
            "training_seconds": float(run["training_seconds"]),
            # log_learning_area drops non-finite steps, so a blow-up must be caught here.
            "diverged": bool(not np.all(np.isfinite(train_losses)) or not np.all(np.isfinite(test_losses))),
        },
        "train_losses": _finite_or_none(train_losses),
        "test_losses": _finite_or_none(test_losses),
    })


def _child_environment(driver_platform, workers):
    environment = dict(os.environ)
    # The driver forced PINN_JAX_PLATFORMS=cpu on itself; the workers get the GPU back.
    environment.pop("JAX_PLATFORMS", None)
    if driver_platform is None:
        environment.pop("PINN_JAX_PLATFORMS", None)
    else:
        environment["PINN_JAX_PLATFORMS"] = driver_platform
    environment.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", f"{0.9 / workers:.3f}")
    return environment


def _describe(job):
    fields = " ".join(f"{field}={value:.4g}" for field, value in job["fields"].items())
    return f"{job['method']:<16} {job['label']:<9} seed={job['seed']} lr={job['learning_rate']:.4g} {fields}"


def _run_jobs(next_job, on_finished, total, epochs, sweep_dir, workers, driver_platform):
    """Train the jobs ``next_job`` returns, ``workers`` at a time, passing each stored run to ``on_finished``."""
    logs_dir = sweep_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    environment = _child_environment(driver_platform, workers)
    running, finished, durations = [], 0, []
    started_at = time.time()

    def launch(job):
        log = (logs_dir / f"{job['path'].stem}.log").open("w")
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--run", job["method"], repr(job["learning_rate"]),
             str(job["seed"]), str(job["path"]), "--fields", json.dumps(job["fields"]), "--epochs", str(epochs)],
            stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=str(HERE),
        )
        print(f"[{finished + len(running) + 1:3d}/{total}] {_describe(job)}  started (pid {process.pid})", flush=True)
        return process, job, log, time.time()

    try:
        while True:
            while len(running) < workers:
                job = next_job()
                if job is None:
                    break
                running.append(launch(job))
            if not running:
                break
            time.sleep(1.0)
            still_running = []
            for entry in running:
                process, job, log, start = entry
                if process.poll() is None:
                    still_running.append(entry)
                    continue
                log.close()
                finished += 1
                elapsed = time.time() - start
                result = _load_run(job["path"], job["config"]) if process.returncode == 0 else None
                if result is None:
                    print(f"[{finished:3d}/{total}] {_describe(job)}  FAILED (exit code {process.returncode}; "
                          f"see {logs_dir / (job['path'].stem + '.log')})", flush=True)
                else:
                    durations.append(elapsed)
                    values = result["metrics"]
                    print(f"[{finished:3d}/{total}] {_describe(job)}  done in {elapsed:4.0f} s: learning area "
                          f"{values['learning_area']:.4f}  final train mse {values['final_train_mse']:.3e}"
                          + ("  DIVERGED" if values["diverged"] else ""), flush=True)
                on_finished(job, result)
                remaining = total - finished
                if durations and remaining > 0:
                    eta = remaining * float(np.mean(durations)) / workers
                    print(f"          {remaining} runs left, about {eta / 60:.0f} min at the current pace", flush=True)
            running = still_running
    except BaseException:
        print("\nstopping the running workers ...", flush=True)
        for process, _, _, _ in running:
            if process.poll() is None:
                process.terminate()
        for process, _, log, _ in running:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            log.close()
        raise
    print(f"{finished} runs trained in {(time.time() - started_at) / 60:.0f} min", flush=True)


def _open_study(storage, train, method, epochs, inherited=None):
    """The stored study of ``method``, or a new one; ``inherited`` are the base method's selected values."""
    import optuna
    from optuna.trial import TrialState

    study = optuna.create_study(storage=storage, study_name=method, direction="minimize", load_if_exists=True)
    # Seeded by the trials so far: a restarted sampler must not repeat its first random draws.
    study.sampler = optuna.samplers.TPESampler(
        multivariate=True, constant_liar=True, n_startup_trials=STARTUP_TRIALS,
        seed=SAMPLER_SEED + 1000 * list(SPACES).index(method) + len(study.trials))
    setup = json.loads(json.dumps({"protocol": _protocol(train, epochs), "search_seeds": list(SEARCH_SEEDS),
                                   "space": [dataclasses.asdict(d) for d in SPACES[method]],
                                   "inherited": inherited}))
    if not study.trials:
        learning_rate, fields = _default_values(train, method)
        params = params_from_values(method, learning_rate, fields)
        outside = [d.name for d in SPACES[method] if not d.low * (1 - 1e-9) <= params[d.name] <= d.high * (1 + 1e-9)]
        if outside:
            raise SystemExit(f"{method}: the starting point lies outside the search box "
                             f"({', '.join(outside)}); widen SPACES")
        for key, value in setup.items():
            study.set_user_attr(key, value)
        study.set_user_attr("incumbent", {"learning_rate": learning_rate, "fields": fields})
        study.enqueue_trial(params, user_attrs={"label": "incumbent"})
        return study
    if {key: study.user_attrs.get(key) for key in setup} != setup:
        raise SystemExit(f"the stored study of {method} used another protocol, search seeds, search box or base "
                         f"values; delete it (optuna delete-study --storage {storage} --study-name {method}) or "
                         "move the sweep directory away to start a new optimization")
    for trial in study.trials:
        if trial.state == TrialState.RUNNING:  # left by an interrupted driver
            study.tell(trial.number, state=TrialState.FAIL)
            study.enqueue_trial(trial.params, user_attrs={"label": trial.user_attrs.get("label", "tpe")})
    return study


def _completed(study):
    from optuna.trial import TrialState

    return [t for t in study.trials if t.state == TrialState.COMPLETE]


def _inherited(base, study):
    """The selected values of the base method ``base``, fixed in every trial of the method searched on top of it."""
    selected = _summarize({base: study})[base]["selected"]
    if selected is None:
        raise SystemExit(f"{base}: no admissible configuration to search on top of")
    return {"method": base, "trial": selected["trial"], "fields": selected["fields"]}


def _fixed_fields(study):
    inherited = study.user_attrs.get("inherited")
    return {} if inherited is None else inherited["fields"]


def _score(per_seed):
    """A trial's search value: its mean learning area, at least DIVERGED_LEARNING_AREA if a seed diverged."""
    areas = [v["learning_area"] for v in per_seed.values() if np.isfinite(v["learning_area"])]
    diverged = any(v["diverged"] or not np.isfinite(v["learning_area"]) for v in per_seed.values())
    score = float(np.mean(areas)) if areas else DIVERGED_LEARNING_AREA
    return max(score, DIVERGED_LEARNING_AREA) if diverged else score


def _jobs(train, method, label, learning_rate, fields, fixed, epochs, sweep_dir):
    """One job per search seed for a configuration; ``fixed`` are the base method's fields, ``fields`` the searched."""
    fields = {**fixed, **fields}
    return {seed: {"method": method, "label": label, "seed": seed, "learning_rate": learning_rate, "fields": fields,
                   "path": _run_path(sweep_dir, method, learning_rate, fields, seed),
                   "config": _run_config(train, method, learning_rate, fields, seed, epochs)}
            for seed in SEARCH_SEEDS}


def _search(train, studies, pending, trials, epochs, sweep_dir, workers, driver_platform, open_study):
    """Complete ``trials`` trials per study, always extending the study with the fewest; every trial
    trains SEARCH_SEEDS and is told once all of its seeds are stored. The ``pending`` methods (their
    base method) are opened with ``open_study`` once their base is complete."""
    import optuna
    from optuna.trial import TrialState

    methods = list(studies)
    complete = {m: len(_completed(studies[m])) for m in methods}
    active, failures = dict.fromkeys(methods, 0), dict.fromkeys(methods, 0)
    distributions = {m: {d.name: optuna.distributions.FloatDistribution(d.low, d.high, log=d.log)
                         for d in SPACES[m]} for m in methods}
    queue, in_flight = [], {}

    def open_pending():
        for method, base in list(pending.items()):
            if complete[base] < trials:
                continue
            inherited = _inherited(base, studies[base])
            studies[method] = open_study(method, inherited)
            del pending[method]
            methods.append(method)
            complete[method], active[method], failures[method] = len(_completed(studies[method])), 0, 0
            distributions[method] = {d.name: optuna.distributions.FloatDistribution(d.low, d.high, log=d.log)
                                     for d in SPACES[method]}
            print(f"          {method}: searched on top of {base} trial {inherited['trial']} "
                  f"({_fields_text(inherited['fields'])})", flush=True)

    def tell(method, trial, results):
        per_seed = {seed: results[seed]["metrics"] for seed in SEARCH_SEEDS}
        trial.set_user_attr("per_seed", _json_safe({str(seed): values for seed, values in per_seed.items()}))
        studies[method].tell(trial, _score(per_seed))
        active[method] -= 1
        complete[method] += 1
        best = studies[method].best_trial
        print(f"          {method}: {complete[method]}/{trials} trials, best learning area "
              f"{best.value:.4f} (trial {best.number})", flush=True)

    def next_job():
        while True:
            if queue:
                return queue.pop(0)
            open_pending()
            open_methods = [m for m in methods if complete[m] + active[m] < trials]
            if not open_methods:
                return None
            method = min(open_methods, key=lambda m: complete[m] + active[m])
            study = studies[method]
            trial = study.ask(distributions[method])
            label = trial.user_attrs.get("label", "tpe")
            if label == "incumbent":
                learning_rate, fields = study.user_attrs["incumbent"]["learning_rate"], study.user_attrs["incumbent"]["fields"]
            else:
                learning_rate, fields = values_from_params(trial.params)
            for key, value in (("label", label), ("learning_rate", learning_rate), ("fields", fields)):
                trial.set_user_attr(key, value)
            jobs = _jobs(train, method, f"trial {trial.number}", learning_rate, fields, _fixed_fields(study), epochs,
                         sweep_dir)
            results = {seed: _load_run(job["path"], job["config"]) for seed, job in jobs.items()}
            active[method] += 1
            if all(run is not None for run in results.values()):
                tell(method, trial, results)
                continue
            missing = [{**job, "trial": trial} for seed, job in jobs.items() if results[seed] is None]
            in_flight[(method, trial.number)] = {"trial": trial, "results": results, "pending": len(missing),
                                                 "failed": False}
            queue.extend(missing)

    def on_finished(job, result):
        method, trial = job["method"], job["trial"]
        key = (method, trial.number)
        state = in_flight[key]
        state["pending"] -= 1
        if result is None:
            if not state["failed"]:
                state["failed"] = True
                studies[method].tell(trial, state=TrialState.FAIL)
                active[method] -= 1
                failures[method] += 1
                if failures[method] > MAX_FAILURES:
                    raise SystemExit(f"{method}: more than {MAX_FAILURES} failed runs; see {sweep_dir / 'logs'}")
                # The trial's other seeds still queued are not worth training.
                queued = [j for j in queue if (j["method"], j["trial"].number) == key]
                for j in queued:
                    queue.remove(j)
                state["pending"] -= len(queued)
        else:
            state["results"][job["seed"]] = result
        if state["pending"] == 0:
            del in_flight[key]
            if not state["failed"]:
                tell(method, trial, state["results"])

    total = (sum(max(0, trials - complete[m]) for m in methods) + trials * len(pending)) * len(SEARCH_SEEDS)
    print(f"\nsearch: {sum(complete.values())} trials already completed, at most {total} runs to train, "
          f"{workers} at a time", flush=True)
    if total:
        _run_jobs(next_job, on_finished, total, epochs, sweep_dir, workers, driver_platform)


def _statistics(per_seed):
    """Mean and population std of every metric over the seeds (nan where a seed lacks it)."""
    arrays = {key: np.array([np.nan if values[key] is None else values[key] for values in per_seed.values()],
                            dtype=float) for key in METRIC_KEYS}
    return ({key: float(np.mean(a)) for key, a in arrays.items()},
            {key: float(np.std(a)) for key, a in arrays.items()})


def _entry(trial):
    per_seed = {int(seed): values for seed, values in trial.user_attrs["per_seed"].items()}
    diverged = sorted(seed for seed, values in per_seed.items() if values["diverged"])
    mean, std = _statistics(per_seed)
    admissible = not diverged and np.isfinite(mean["learning_area"])
    return {"trial": trial.number, "label": trial.user_attrs["label"], "value": trial.value,
            "learning_rate": trial.user_attrs["learning_rate"], "fields": trial.user_attrs["fields"],
            "params": trial.params, "per_seed": per_seed, "diverged_seeds": diverged, "mean": mean, "std": std,
            "score": mean["learning_area"] if admissible else float("inf")}


def _summarize(studies):
    """Per method: every completed trial, best first, and the selected one."""
    summary = {}
    for method, study in studies.items():
        entries = sorted((_entry(t) for t in _completed(study)), key=lambda e: (e["score"], e["value"], e["trial"]))
        admissible = [e for e in entries if np.isfinite(e["score"])]
        selected = admissible[0] if admissible else None
        edge = []
        if selected is not None:
            params = params_from_values(method, selected["learning_rate"], selected["fields"])
            edge = [d.name for d in SPACES[method] if not 0.02 <= _unit(d, params[d.name]) <= 0.98]
        summary[method] = {"entries": entries, "selected": selected, "near_box_edge": edge,
                           "incumbent": next((e for e in entries if e["label"] == "incumbent"), None),
                           "inherited": study.user_attrs.get("inherited")}
    return summary


def _shown(train, method, entry, fixed, epochs, sweep_dir):
    """The stored runs of ``entry``, one per seed (None where missing)."""
    jobs = _jobs(train, method, f"trial {entry['trial']}", entry["learning_rate"], entry["fields"], fixed, epochs,
                 sweep_dir)
    return {seed: _load_run(job["path"], job["config"]) for seed, job in jobs.items()}


def _constant_name(method):
    """optimizer_config's METHOD_* name for ``method``, for a readable key."""
    for name, value in vars(optimizer_config).items():
        if name.startswith("METHOD_") and value == method:
            return f"optimizer_config.{name}"
    return repr(method)


def _marker_span(lines, begin, end, exact):
    matches = (lambda line, marker: line.strip() == marker) if exact else (
        lambda line, marker: line.strip().startswith(marker))
    begins = [i for i, line in enumerate(lines) if matches(line, begin)]
    ends = [i for i, line in enumerate(lines) if matches(line, end)]
    if len(begins) != 1 or len(ends) != 1 or begins[0] >= ends[0]:
        raise RuntimeError(f"train.py must contain exactly one {begin!r} line before one {end!r} line")
    indent = lines[begins[0]][: len(lines[begins[0]]) - len(lines[begins[0]].lstrip())]
    return begins[0], ends[0], indent


def write_selection(train_path, train, epochs, trials, summary):
    """Rewrite train.py's learning rates and every method's tuned fields between their markers."""
    train_path = Path(train_path)
    lines = train_path.read_text(encoding="utf-8").splitlines(keepends=True)
    stamp = (f"# {datetime.datetime.now().isoformat(timespec='minutes')}: sweep_hpo.py, {epochs} epochs, TPE "
             f"{trials} trials per method on seeds {SEARCH_SEEDS},\n",
             "# criterion: lowest mean learning_area over the seeds.\n")

    begin, end, indent = _marker_span(lines, *LR_MARKERS, exact=False)
    body = [indent + line for line in stamp]
    for method in optimizer_config.METHODS:
        selected = summary[method]["selected"] if method in summary else None
        if selected is not None:
            body.append(f"{indent}{_constant_name(method)}: {selected['learning_rate']!r},  # learning_area "
                        f"{selected['mean']['learning_area']:.4f} +- {selected['std']['learning_area']:.4f}\n")
        elif method in train.LEARNING_RATE_OVERRIDES:
            note = "  # no admissible configuration: kept" if method in summary else ""
            body.append(f"{indent}{_constant_name(method)}: {train.LEARNING_RATE_OVERRIDES[method]!r},{note}\n")
    lines = lines[: begin + 1] + body + lines[end:]

    for method, entry in summary.items():
        if entry["selected"] is None:
            continue
        begin, end, indent = _marker_span(lines, f"# BEGIN HPO {method}", f"# END HPO {method}", exact=True)
        current = _config(train, method)
        body = [indent + line for line in stamp]
        for field in _field_names(train, method):
            old, new = getattr(current, field), entry["selected"]["fields"][field]
            was = f"  # was {old:g}" if new != old else ""
            body.append(f"{indent}{field}={new!r},{was}\n")
        lines = lines[: begin + 1] + body + lines[end:]

    new_text = "".join(lines)
    ast.parse(new_text, filename=str(train_path))  # never leave train.py unparseable
    temporary = train_path.with_name(f"{train_path.name}.{os.getpid()}.part")
    temporary.write_text(new_text, encoding="utf-8")
    os.replace(temporary, train_path)


def _importance(study):
    import optuna

    if len(_completed(study)) < 2 * len(SPACES[study.study_name]):
        return {}
    try:
        importances = optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.PedAnovaImportanceEvaluator())
    except Exception:  # an importance estimate must never cost the reports
        return {}
    return {name: float(value) for name, value in importances.items()}


def _format(value, spec=".4f"):
    return "—" if value is None or not np.isfinite(value) else format(value, spec)


def _fields_text(fields):
    return ", ".join(f"{field} {value:.4g}" for field, value in fields.items())


def _entry_json(entry):
    return {**entry, "per_seed": {str(seed): values for seed, values in entry["per_seed"].items()}}


def _write_json(path, train, studies, epochs, trials, summary, wrote_train_py):
    payload = {
        "case": "4-PINN-opt",
        **_protocol(train, epochs),
        "trials_per_method": trials,
        "search_seeds": list(SEARCH_SEEDS),
        "startup_trials": STARTUP_TRIALS,
        "sampler_seed": SAMPLER_SEED,
        "criterion": "learning_area",
        "written_to_train_py": wrote_train_py,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "methods": {
            method: {
                "space": [dataclasses.asdict(d) for d in SPACES[method]],
                "starting_point": study.user_attrs["incumbent"],
                "inherited": study.user_attrs.get("inherited"),
                "importance": _importance(study),
                "selected": None if summary[method]["selected"] is None else _entry_json(summary[method]["selected"]),
                "near_box_edge": summary[method]["near_box_edge"],
                "trials": [_entry_json(entry) for entry in summary[method]["entries"]],
            }
            for method, study in studies.items()
        },
    }
    _write_json_atomic(path, _json_safe(payload))


def _write_markdown(path, train, studies, epochs, trials, summary):
    lines = [
        "# Joint hyperparameter optimization, 4-PINN-opt",
        "",
        f"{epochs} epochs on the full batch of {train.N_TRAIN} boundary points ({train.N_TEST} test points), "
        f"layers {tuple(train.model.LAYER_SIZES)}. Search: Optuna TPE (multivariate), {trials} trials per method, "
        f"every trial trained on the weight-initialization seeds {SEARCH_SEEDS} (boundary sampling seed "
        f"{train.BOUNDARY_SAMPLING_SEEDS[0]}), the first at the starting point: learning rate {train.LEARNING_RATE:g} "
        "and the defaults of src/optimizer_config.py. "
        + "; ".join(f"{m} is searched once {b} is complete, on top of its selected values" for m, b in BASES.items())
        + ".",
        "Selection: lowest mean learning area over the seeds; a trial that diverged on any seed is excluded.",
        f"Generated {datetime.datetime.now().isoformat(timespec='minutes')}.",
        "",
        "## Selected configurations",
        "",
        "| method | learning rate | tuned fields | learning area (mean ± std) | starting point | note |",
        "|---|---|---|---|---|---|",
    ]
    for method, entry in summary.items():
        selected, incumbent = entry["selected"], entry["incumbent"]
        start = (f"{incumbent['mean']['learning_area']:.4f} ± {incumbent['std']['learning_area']:.4f}"
                    if incumbent is not None and np.isfinite(incumbent["score"]) else "—")
        if selected is None:
            lines.append(f"| {method} | — | — | — | {start} | no admissible configuration |")
            continue
        note = "starting point retained" if selected["label"] == "incumbent" else f"trial {selected['trial']}"
        if entry["inherited"] is not None:
            note += f"; on top of {entry['inherited']['method']} trial {entry['inherited']['trial']}"
        if entry["near_box_edge"]:
            note += f"; near the box edge: {', '.join(entry['near_box_edge'])}"
        lines.append(f"| {method} | {selected['learning_rate']:.4g} | {_fields_text(selected['fields'])} | "
                     f"{selected['mean']['learning_area']:.4f} ± {selected['std']['learning_area']:.4f} | "
                     f"{start} | {note} |")

    for method, study in studies.items():
        entry = summary[method]
        importance = ", ".join(f"{name} {value:.2f}" for name, value in _importance(study).items())
        lines += [
            "", f"## {method}", "",
            "Search box: " + ", ".join(f"{d.name} [{d.low:g}, {d.high:g}]{' (log)' if d.log else ''}"
                                       for d in SPACES[method]) + ".",
            *([f"On top of {entry['inherited']['method']} trial {entry['inherited']['trial']}: "
               f"{_fields_text(entry['inherited']['fields'])}."] if entry["inherited"] is not None else []),
            f"Importance (PedANOVA): {importance or '—'}.",
            "", f"### Trials, best first ({len(entry['entries'])} completed)", "",
            "| trial | learning area (mean ± std) | " + " | ".join(f"seed {s}" for s in SEARCH_SEEDS)
            + " | learning rate | tuned fields | final train MSE | final test MSE | diverged |",
            "|---|---|" + "---|" * len(SEARCH_SEEDS) + "---|---|---|---|---|",
        ]
        for e in entry["entries"]:
            mark = " **←**" if entry["selected"] is e else ""
            name = f"{e['trial']} (start)" if e["label"] == "incumbent" else str(e["trial"])
            per_seed = " | ".join(_format(e["per_seed"][s]["learning_area"]) if s in e["per_seed"] else "—"
                                  for s in SEARCH_SEEDS)
            lines.append(f"| {name}{mark} | {_format(e['mean']['learning_area'])} ± {_format(e['std']['learning_area'])} "
                         f"| {per_seed} | {e['learning_rate']:.4g} | {_fields_text(e['fields'])} | "
                         f"{_format(e['mean']['final_train_mse'], '.3e')} | {_format(e['mean']['final_test_mse'], '.3e')} "
                         f"| {', '.join(map(str, e['diverged_seeds'])) or '—'} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _block_mean(curve, width):
    """Mean over consecutive windows of ``width`` epochs, with the window centres."""
    starts = np.arange(0, curve.size, width)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows of diverged runs
        means = np.array([np.nanmean(curve[a:a + width]) for a in starts])
    centres = starts + np.minimum(width, curve.size - starts) / 2
    return centres, means


def _write_pdf(path, train, studies, summary, epochs, sweep_dir):
    from matplotlib.backends.backend_pdf import PdfPages

    from src import comparison_report

    plt = comparison_report._pyplot(fontsize=8)

    def label(text):
        return text.replace("_", r"\_") if comparison_report.usetex_enabled() else text

    methods = list(studies)
    columns = 4
    rows = max(1, -(-len(methods) // columns))

    def grid_of_axes(title):
        figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.0 * rows), squeeze=False)
        figure.suptitle(title)
        for axis in axes.ravel()[len(methods):]:
            axis.set_visible(False)
        return figure, axes.ravel()

    def clip_top(axis, values):
        if values.size:
            low, top = float(values.min()), float(np.quantile(values, 0.9))
            margin = 0.05 * max(top - low, 1e-3)
            axis.set_ylim(low - margin, top + margin)

    with PdfPages(path) as pdf:
        figure, axes = grid_of_axes(
            f"4-PINN-opt: mean learning area over seeds {SEARCH_SEEDS} of every trial (x marks a trial with a "
            "diverged seed) and the lowest so far; values above the 90th percentile are clipped")
        for axis, method in zip(axes, methods):
            entries = sorted(summary[method]["entries"], key=lambda e: e["trial"])
            if not entries:
                continue
            color = optimizer_config.METHOD_COLORS.get(method, "black")
            numbers = np.array([e["trial"] for e in entries])
            values = np.array([e["value"] for e in entries])
            diverged = np.array([bool(e["diverged_seeds"]) for e in entries])
            axis.plot(numbers[~diverged], values[~diverged], "o", color=color, markersize=3, label="trial")
            if diverged.any():
                axis.plot(numbers[diverged], values[diverged], "x", color=color, markersize=5, label="diverged")
            axis.step(numbers, np.minimum.accumulate(values), where="post", color="black", linewidth=1.0,
                      label="lowest so far")
            clip_top(axis, values)
            axis.set_title(comparison_report._latex_method(method))
            axis.set_xlabel("trial")
            axis.set_ylabel("learning area")
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=6)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(figure)
        plt.close(figure)

        width = max(len(SPACES[method]) for method in methods)
        figure, grid = plt.subplots(len(methods), width, figsize=(2.8 * width, 2.3 * len(methods)), squeeze=False)
        figure.suptitle(f"4-PINN-opt: mean learning area over seeds {SEARCH_SEEDS} against each search parameter, "
                        "over its box; the circle marks the selected configuration")
        for row, method in zip(grid, methods):
            color = optimizer_config.METHOD_COLORS.get(method, "black")
            entries = [e for e in summary[method]["entries"] if np.isfinite(e["score"])]
            values = np.array([e["value"] for e in entries])
            selected = summary[method]["selected"]
            for axis, dimension in zip(row, SPACES[method]):
                axis.plot([e["params"][dimension.name] for e in entries], values, "o", color=color, markersize=3)
                if selected is not None:
                    chosen = params_from_values(method, selected["learning_rate"], selected["fields"])
                    axis.plot(chosen[dimension.name], selected["value"], "o",
                              markerfacecolor="none", markeredgecolor="black", markersize=10)
                axis.set_xlim(dimension.low, dimension.high)
                if dimension.log:
                    axis.set_xscale("log")
                clip_top(axis, values)
                axis.set_title(label(f"{method}: {dimension.name}"), fontsize=7)
                axis.grid(True, which="both", alpha=0.3)
            for axis in row[len(SPACES[method]):]:
                axis.set_visible(False)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = grid_of_axes(
            f"4-PINN-opt: training loss of the {SHOWN_TRIALS} best trials and the starting point, mean over "
            f"windows of {CURVE_WINDOW} epochs and over seeds {SEARCH_SEEDS}")
        for axis, method in zip(axes, methods):
            entry = summary[method]
            color = optimizer_config.METHOD_COLORS.get(method, "black")
            shown = [e for e in entry["entries"] if np.isfinite(e["score"])][:SHOWN_TRIALS]
            if entry["incumbent"] is not None and entry["incumbent"] not in shown:
                shown.append(entry["incumbent"])
            for style, e in zip(("-", "--", ":", "-."), shown):
                runs = [run for run in _shown(train, method, e, _fixed_fields(studies[method]), epochs,
                                              sweep_dir).values() if run is not None]
                if not runs:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns of diverged runs
                    mean = np.nanmean(np.stack([_from_none(run["train_losses"]) for run in runs]), axis=0)
                centres, mean = _block_mean(mean, CURVE_WINDOW)
                chosen = entry["selected"] is e
                name = f"trial {e['trial']}" + (" (start)" if e["label"] == "incumbent" else "") + (
                    " (selected)" if chosen else "")
                axis.plot(centres, mean, style, color=color if chosen else "0.45", linewidth=1.4 if chosen else 0.9,
                          label=label(name))
            axis.set_yscale("log")
            axis.set_title(comparison_report._latex_method(method))
            axis.set_xlabel("epoch")
            axis.set_ylabel("training loss")
            axis.grid(True, which="both", alpha=0.3)
            axis.legend(fontsize=6)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(figure)
        plt.close(figure)


def _parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", nargs=4, metavar=("METHOD", "LR", "SEED", "OUT"), help="internal: train one run in this process")
    parser.add_argument("--fields", default=None, help="internal: the run's config fields as JSON")
    parser.add_argument("--epochs", type=int, default=None, help="epoch budget; default train.N_EPOCHS (override for smoke tests only)")
    parser.add_argument("--trials", type=int, default=TRIALS, help=f"completed trials per method (default {TRIALS}; override for smoke tests only)")
    parser.add_argument("--methods", nargs="+", default=list(optimizer_config.METHODS), help="methods to optimize (default optimizer_config.METHODS)")
    parser.add_argument("--report-only", action="store_true", help="skip training; report from the stored study and runs")
    parser.add_argument("--no-write", action="store_true", help="do not write the selection into train.py")
    parser.add_argument("--results-dir", type=Path, default=None, help="sweep directory; default <train.RESULTS_DIR>/hpo")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)

    if arguments.run:
        method, learning_rate, seed, out_path = arguments.run
        _worker(method, float(learning_rate), json.loads(arguments.fields), int(seed), arguments.epochs, Path(out_path))
        return 0

    # Keep the driver off the GPU; train.py reads it at import to select the JAX backend.
    driver_platform = os.environ.get("PINN_JAX_PLATFORMS")
    os.environ["PINN_JAX_PLATFORMS"] = "cpu"
    import optuna

    import train

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    unknown = [m for m in arguments.methods if m not in SPACES]
    if unknown:
        raise SystemExit(f"no search space for {unknown}; SPACES covers {tuple(SPACES)}")
    epochs = arguments.epochs or train.N_EPOCHS
    sweep_dir = arguments.results_dir or (Path(train.RESULTS_DIR) / SWEEP_DIRNAME)
    (sweep_dir / "runs").mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{sweep_dir / 'study.db'}"
    studies = {method: _open_study(storage, train, method, epochs) for method in arguments.methods
               if method not in BASES}
    # The AURA methods wait for their base method; a study stored before the base was complete is stale.
    pending, stored = {}, set(optuna.study.get_all_study_names(storage))
    for method in [m for m in arguments.methods if m in BASES]:
        base = BASES[method]
        if base not in studies:
            raise SystemExit(f"{method} is searched on top of {base}: add {base} to --methods")
        if len(_completed(studies[base])) >= arguments.trials:
            studies[method] = _open_study(storage, train, method, epochs, _inherited(base, studies[base]))
        elif method in stored:
            raise SystemExit(f"the stored study of {method} predates the completion of {base}; delete it "
                             f"(optuna delete-study --storage {storage} --study-name {method}) or move the sweep "
                             "directory away")
        else:
            pending[method] = base

    print(f"4-PINN-opt joint optimization: {epochs} epochs on {train.N_TRAIN} boundary points, TPE "
          f"{arguments.trials} trials per method, every trial trained on seeds {SEARCH_SEEDS}\n"
          f"  results under {sweep_dir}", flush=True)
    for method in arguments.methods:
        print(f"  {method}: " + ", ".join(f"{d.name} [{d.low:g}, {d.high:g}]{' log' if d.log else ''}"
                                          for d in SPACES[method])
              + (f"; searched once {BASES[method]} is complete, on top of its selected values" if method in BASES
                 else ""), flush=True)
        if method not in studies:
            continue
        learning_rate, fields = _default_values(train, method)
        start = json.loads(json.dumps({"learning_rate": learning_rate, "fields": fields}))
        if start != studies[method].user_attrs["incumbent"]:
            print(f"    note: the defaults differ from the starting point of the stored study of {method}", flush=True)

    if not arguments.report_only:
        _search(train, studies, pending, arguments.trials, epochs, sweep_dir, WORKERS, driver_platform,
                lambda method, inherited: _open_study(storage, train, method, epochs, inherited))
    studies = {method: studies[method] for method in arguments.methods if method in studies}

    summary = _summarize(studies)
    searched = not pending and all(len(_completed(study)) >= arguments.trials for study in studies.values())

    print(f"\nSelected configurations (lowest mean learning area over seeds {SEARCH_SEEDS}):\n", flush=True)
    for method, entry in summary.items():
        selected, incumbent = entry["selected"], entry["incumbent"]
        start = (f"starting point {incumbent['mean']['learning_area']:.4f}"
                    if incumbent is not None and np.isfinite(incumbent["score"]) else "starting point -")
        if selected is None:
            print(f"{method:<16} no admissible configuration; {start}")
            continue
        fields = " ".join(f"{field}={value:.4g}" for field, value in selected["fields"].items())
        edge = f"  near the box edge: {', '.join(entry['near_box_edge'])}" if entry["near_box_edge"] else ""
        on_top = (f"; on top of {entry['inherited']['method']} trial {entry['inherited']['trial']}"
                  if entry["inherited"] is not None else "")
        print(f"{method:<16} lr={selected['learning_rate']:<8.4g} {fields:<66} learning area "
              f"{selected['mean']['learning_area']:.4f} +- {selected['std']['learning_area']:.4f} "
              f"({start}; trial {selected['trial']}{on_top}){edge}")

    wrote = False
    if arguments.no_write:
        print("\n--no-write: train.py left unchanged")
    elif not searched:
        print("\ntrain.py left unchanged: the optimization is incomplete (rerun to finish it)")
    else:
        write_selection(HERE / "train.py", train, epochs, arguments.trials, summary)
        wrote = True
        print(f"\nwrote the selected configurations into {HERE / 'train.py'}")

    _write_json(sweep_dir / "hpo.json", train, studies, epochs, arguments.trials, summary, wrote)
    _write_markdown(sweep_dir / "hpo.md", train, studies, epochs, arguments.trials, summary)
    try:
        _write_pdf(sweep_dir / "hpo.pdf", train, studies, summary, epochs, sweep_dir)
    except Exception:  # a figure must never cost the selection
        traceback.print_exc()
        print("!!! could not draw hpo.pdf; the JSON and Markdown reports are complete")
    print(f"reports: {sweep_dir / 'hpo.json'}, {sweep_dir / 'hpo.md'}, {sweep_dir / 'hpo.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
