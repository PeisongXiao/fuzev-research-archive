"""Render the six comparison charts from published aggregate result JSON."""

import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


BLUE, PURPLE, LIGHT_PURPLE = "#2563eb", "#7c3aed", "#b9a0eb"
GREY, DARK_GREY, LIGHT_GREY = "#92969d", "#555960", "#c9ccd0"
DATASET_NAMES = {"apibank": "API-Bank", "bfcl": "BFCL", "bright": "BRIGHT",
    "cfcolor": "cfcolor", "chessbench": "ChessBench", "clinc150": "CLINC150",
    "contractnli": "ContractNLI", "hellaswag": "HellaSwag", "humicroedit": "Humicroedit",
    "musr": "MuSR", "routerbench": "RouterBench", "sata_bench": "SATA-Bench",
    "sgd": "SGD / SGD-X", "toolret": "ToolRet"}
SPEED_ARMS = [
    ("speed-v1.0.json", BLUE, "-", "o"),
    ("speed-v1.1.json", PURPLE, "-", "o"),
    ("speed-exllamav3-1.6.2-fp32.json", GREY, "-", "s"),
    ("speed-exllamav3-1.6.2.json", GREY, "--", "s"),
    ("speed-exllamav3-master-fp32.json", DARK_GREY, "-", "^"),
    ("speed-exllamav3-master.json", DARK_GREY, "--", "^"),
]
FUZEV_ENTRIES = [
    (BLUE, "FuZev v1.0: Qwen3.8-Flash-Next EXL3 4.05 bpw + head 1.0.0 (Hub tag v1.0), raw"),
    (PURPLE, "FuZev v1.1: same backbone + head 1.1.0 (Hub tag v1.1), calibrated"),
]


def read(results, name):
    return json.loads((results / name).read_text())


def save(fig, out, name, notes, *, groups=()):
    """Write PNG and SVG with titled legend groups side by side above a wrapped footer."""
    lines = [line for note in notes for line in textwrap.wrap(note, width=160)]
    height = fig.get_figheight()
    footer_height = (len(lines) * 0.15 + 0.18) / height
    rows = max((len(labels) for _, _, labels in groups), default=0)
    legend_height = (0.4 + 0.19 * rows) / height if groups else 0
    fig.tight_layout(rect=(0, footer_height + legend_height, 1, 0.95))
    for index, (title, handles, labels) in enumerate(groups):
        fig.legend(handles, labels, title=title, loc="upper left", alignment="left",
                   bbox_to_anchor=(0.04 + index * 0.48, footer_height + legend_height),
                   frameon=False, fontsize=7.5, title_fontsize=8)
    fig.text(0.015, 0.018, "\n".join(lines), va="bottom", fontsize=7, color="#4b4b4b",
             linespacing=1.3)
    for extension in ("png", "svg"):
        fig.savefig(out / f"{name}.{extension}", dpi=160)
    plt.close(fig)


def decision_notes(record, reference):
    uncertainty = record["uncertainty"]
    return [f"{reference['label']} breadth-v1 from {reference['version']}; the same scorer for every system; "
            f"{uncertainty['samples']:,} whole-record resamples within each dataset, seed {uncertainty['seed']}.",
            record["measurement_note"]]


def fuzev_group():
    return ("FuZev", [Patch(color=color) for color, _ in FUZEV_ENTRIES], [label for _, label in FUZEV_ENTRIES])


def highlight_fuzev(ax):
    for label in ax.get_yticklabels():
        if label.get_text().startswith("FuZev"):
            label.set_color(BLUE if label.get_text().endswith("1.0") else PURPLE)
            label.set_fontweight("bold")


def index_chart(results, out, reference):
    names = ["Kev-0.8B", "Kev-4B", "Kev-9B v2", "FuZev v1.0", "FuZev v1.1", "Kev-27B v2", "Jev"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    for ax, split, public_split in zip(axes, ("dev", "test"), ("development", "test")):
        record = read(results, f"kev-{split}-v1.1-vs-v1.0.json")
        values = []
        for index, name in enumerate(names):
            if name.startswith("FuZev"):
                value = record["systems"][name]["overall"]["index"]
                low, high = record["uncertainty"]["index"][name]
                color = BLUE if name.endswith("1.0") else PURPLE
            else:
                ref = reference["systems"][name][public_split]
                value = ref["documented_aggregate"]["overall"]["index"]
                low, high = ref["index_ci95"]
                color = DARK_GREY if name == "Jev" else LIGHT_GREY
            ax.barh(index, value, color=color, height=0.62)
            ax.errorbar(value, index, xerr=[[value - low], [high - value]],
                        fmt="none", ecolor="#222222", capsize=3, lw=1)
            fuzev = name.startswith("FuZev")
            ax.text(high + 0.6, index, f"{value:.2f}", va="center", fontsize=8,
                    color=color if fuzev else "black", fontweight="bold" if fuzev else "normal")
            values.append(high)
        ax.set_yticks(range(len(names)), ["Jev (hosted)" if name == "Jev" else name for name in names])
        highlight_fuzev(ax)
        ax.set_xlim(0, max(values) * 1.16)
        ax.set_title(f"{split.upper()} ({record['questions']:,} questions)")
        ax.set_xlabel("breadth-v1 decision index (95% bootstrap CI)")
    fig.suptitle("Decision Quality on Kev breadth-v1: FuZev, the Kev Family and Hosted Jev", fontweight="bold")
    references = ("References", [Patch(color=LIGHT_GREY), Patch(color=DARK_GREY)],
                  ["Kev family (0.8B, 4B, 9B v2, 27B v2)", "Jev (hosted)"])
    save(fig, out, "kev-index", decision_notes(record, reference), groups=[fuzev_group(), references])


def area_chart(results, out, reference):
    record = read(results, "kev-test-v1.1-vs-v1.0.json")
    names = ["Kev-0.8B", "Kev-4B", "Kev-9B v2", "FuZev v1.0", "FuZev v1.1", "Kev-27B v2", "Jev"]
    colors = ["#dedfe1", LIGHT_GREY, GREY, BLUE, PURPLE, "#73777e", DARK_GREY]
    areas = record["systems"]["FuZev v1.1"]["areas"]
    fig, ax = plt.subplots(figsize=(14, 7))
    x, width = np.arange(len(areas)), 0.115
    for index, (name, color) in enumerate(zip(names, colors)):
        source = record["systems"][name] if name.startswith("FuZev") else reference["systems"][name]["test"]["documented_aggregate"]
        values = [100 * source["areas"][key]["skill"] for key in areas]
        bars = ax.bar(x + (index - 3) * width, values, width, color=color, label=name)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=6)
    ax.set_xticks(x, [textwrap.fill(area["label"], 20) for area in areas.values()])
    ax.set_ylim(bottom=0, top=max(p.get_height() for p in ax.patches) * 1.15)
    ax.set_ylabel("chance-corrected skill x 100")
    fig.suptitle("TEST by Area: The Index Is the Mean of Five Areas", fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    others = [(handle, "Jev (hosted)" if label == "Jev" else label)
              for handle, label in zip(handles, labels) if not label.startswith("FuZev")]
    save(fig, out, "kev-areas", decision_notes(record, reference) +
         ["Kev publishes no TEST prediction rows or per-area intervals; area bars carry no whiskers."],
         groups=[fuzev_group(), ("References", [h for h, _ in others], [label for _, label in others])])


def delta_chart(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    for ax, split in zip(axes, ("dev", "test")):
        record = read(results, f"kev-{split}-v1.1-vs-v1.0.json")
        base, current = (record["systems"][name] for name in ("FuZev v1.0", "FuZev v1.1"))
        names = sorted(base["datasets"], key=lambda key: DATASET_NAMES[key])
        values = [100 * (current["datasets"][name]["score"] - base["datasets"][name]["score"]) for name in names]
        intervals = [record["uncertainty"]["dataset_difference"][name]["FuZev v1.1"] for name in names]
        y = np.arange(len(names))
        ax.barh(y, values, color=PURPLE, alpha=0.8, height=0.55)
        ax.errorbar(values, y, xerr=[[value - ci[0] for value, ci in zip(values, intervals)],
            [ci[1] - value for value, ci in zip(values, intervals)]], fmt="none", ecolor="#222222", capsize=3)
        for row, value, ci in zip(y, values, intervals):
            label = Decimal(f"{value:.9f}").quantize(Decimal("0.01"), ROUND_HALF_UP)
            ax.annotate(f"{label:+}", (max(ci[1], 0), row), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=7)
        labels = [DATASET_NAMES[name] + (" (case-exact)" if base["datasets"][name]["metric"] == "case_exact" else "") for name in names]
        ax.set_yticks(y, labels)
        ax.axvline(0, color="#222222", lw=0.8)
        low, high = record["uncertainty"]["difference"]["FuZev v1.1"]
        delta = current["overall"]["index"] - base["overall"]["index"]
        ax.set_title(f"{split.upper()} index difference: {delta:+.2f} [{low:+.2f}, {high:+.2f}]")
        ax.set_xlabel("dataset score change (percentage points)")
        ax.set_xlim(min(ci[0] for ci in intervals) - 2, max(ci[1] for ci in intervals) + 5)
    fig.suptitle("FuZev v1.1 Minus v1.0: Paired 95% Intervals", fontweight="bold")
    save(fig, out, "text-deltas", [
        f"Dataset score change, FuZev v1.1 minus v1.0 (percentage points). {record['uncertainty']['samples']:,} "
        f"whole-record resamples within each dataset, seed {record['uncertainty']['seed']}. "
        "BFCL and SATA-Bench use case-exact scores. Overall unchanged from v1.0 within the CI.",
        record["measurement_note"]])


def image_chart(results, out):
    records = [read(results, name) for name in ("vision-v1.0.json", "vision-v1.1-v1-head.json", "vision-v1.1.json")]
    tracks = [("eval-mix", "mm-rlhf/choice/image/best_response", "MM-RLHF\nbest response"),
        ("eval-mix", "mm-rlhf/choice/image/pairwise_preference", "MM-RLHF\npairwise"),
        ("eval-mix", "vl-rewardbench/choice/image/official_heldout_preference", "VL-RewardBench\nChoice (mix)"),
        ("eval-mix", "vl-rewardbench/noul/image/pairwise_preference_yes_no", "VL-RewardBench\nNoul (mix)"),
        ("vl-choice", "vl-rewardbench/choice/image/official_heldout_preference", "VL-RewardBench\nChoice (full)"),
        ("vl-noul", "vl-rewardbench/noul/image/pairwise_preference_yes_no", "VL-RewardBench\nNoul (full)"),
        ("attribute-eval", "sugarcrepe/choice/image/replace_att", "SugarCrepe\nreplace-att EVAL")]
    arms = [(records[0], "head", "FuZev v1.0 (head 1.0.0)", BLUE),
            (records[1], "head", "FuZev v1.1 engine with head 1.0.0 (control)", LIGHT_PURPLE),
            (records[2], "head", "FuZev v1.1 (head 1.1.0, calibrated)", PURPLE),
            (records[2], "backbone", "Frozen-LM label control", GREY)]
    def entry(record, method, name, track):
        return next(row for row in record["retry_completed"] if
                    (row["method"], row["set"], row["track"]) == (method, name, track))
    fig, ax = plt.subplots(figsize=(13, 9))
    x, width = np.arange(len(tracks)), 0.2
    counts = []
    for index, (record, method, label, color) in enumerate(arms):
        entries = [entry(record, method, name, track) for name, track, _ in tracks]
        values = [100 * row["metrics"]["accuracy"] for row in entries]
        bars = ax.bar(x + (index - 1.5) * width, values, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=6)
        choice, noul = (entry(record, method, name, track) for name, track, _ in tracks[4:6])
        counts.append(f"{label}: {choice['gold_second_correct']}/{choice['gold_second']} Choice, "
                      f"{noul['gold_second_correct']}/{noul['gold_second']} Noul")
    ax.set_xticks(x, [f"{label}\n({entry(records[2], 'head', name, track)['scored']:,} scored)"
                       for name, track, label in tracks], fontsize=7)
    ax.set_ylim(0, max(p.get_height() for p in ax.patches) * 1.13)
    ax.set_ylabel("accuracy (%)")
    fig.suptitle("Image Decisions (Experimental)", fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    full = entry(records[2], "head", tracks[4][0], tracks[4][1])
    balance = f"{full['gold_first']:,} of {full['requests']:,} ({100 * full['gold_first'] / full['requests']:.1f}%)"
    save(fig, out, "image-decisions", [
        f"Mix: {entry(records[2], 'head', 'eval-mix', 'all')['requests']:,} requests. "
        "Accuracy counts any equally best gold option as correct. "
        "Requests that returned HTTP 413 at the default 8 MiB request limit were re-sent once as lossless WebP; "
        f"{len(full['errors'])} requests per full VL-RewardBench track still fail and are unscored.",
        f"VL-RewardBench: preferred response first in {balance}; accuracy mostly reflects how often a system "
        "answers Yes or picks the first response. Correct where the second response is preferred: " + "; ".join(counts) + ".",
        "SugarCrepe replace-att pairs COCO val2017 images with correct and attribute-swapped captions. "
        "Fit/development selected the image residual strength and temperature; the evaluation views are held out from both.",
        records[2]["measurement_note"]], groups=[("FuZev", handles[:3], labels[:3]),
                                                 ("References", handles[3:], labels[3:])])


def speed_notes(records, phase):
    envs = [record["environment"] for record in records]
    shared = ("device", "device_memory_description", "python", "torch", "triton", "flashinfer_python",
              "max_batch_size", "kv_cache_bits")
    if all(len({str(env.get(key)) for env in envs}) == 1 for key in shared):
        env = envs[0]
        software = ", ".join(f"{label} {env[key]}" for key, label in
                             (("python", "Python"), ("torch", "PyTorch"), ("triton", "Triton"),
                              ("flashinfer_python", "FlashInfer")) if key in env)
        upstream = sorted({env["engine"] for env in envs if not env["engine"].startswith("exllamav3-fuzev")})
        chunks = sorted({record["generator_options"]["max_chunk_size"] for record in records})
        return [f"All speeds measured on one {env['device']} ({env['device_memory_description']}); {software}.",
                "Upstream builds: " + "; ".join(upstream) + ".",
                f"Batch {env['max_batch_size']}, KV cache bits {'/'.join(map(str, env['kv_cache_bits']))}, "
                "prefill chunk " + "/".join(f"{chunk:,}" for chunk in chunks) +
                "; FP32/FP16 = recurrent-state precision.",
                "MTP draft component loaded; prefill rows generate one token." if phase == "C" else
                "MTP 3 and MTP off. MTP-off panel uses a zoomed y-axis. Temperature 1.0; MTP differences "
                "combine speed and draft acceptance."]
    notes = []
    for record in records:
        env = record["environment"]
        memory = env.get("device_memory_description") or f"{env['device_memory_bytes'] / 2**30:.1f} GiB"
        parts = [f"{record['label']}: one {env['device']} ({memory})"]
        for key, label in (("python", "Python"), ("torch", "PyTorch"), ("triton", "Triton"),
                           ("flashinfer_python", "FlashInfer"), ("flashinfer_jit_cache", "kernel cache")):
            if key in env:
                parts.append(f"{label} {env[key]}")
        parts += [env.get("engine_build", env["engine"]), f"batch {env['max_batch_size']}",
                  "KV cache bits " + "/".join(map(str, env["kv_cache_bits"])),
                  f"chunk {record['generator_options']['max_chunk_size']:,}",
                  "recurrent state " + "/".join(env["recurrent_state"]).replace("torch.float", "FP")]
        notes.append(", ".join(parts) + ".")
    notes.append("MTP draft component loaded; prefill rows generate one token." if phase == "C" else
                 "MTP 3 and MTP off. MTP-off panel uses a zoomed y-axis. Temperature 1.0; MTP differences combine speed and draft acceptance.")
    return notes


def speed_groups(handles, labels):
    fuzev = [index for index, label in enumerate(labels) if label.startswith("FuZev")]
    upstream = [index for index in range(len(labels)) if index not in fuzev]
    return [("FuZev", [handles[i] for i in fuzev], [labels[i] for i in fuzev]),
            ("Upstream ExLlamaV3", [handles[i] for i in upstream], [labels[i] for i in upstream])]


def speed_charts(results, out):
    records = [read(results, spec[0]) for spec in SPEED_ARMS]
    data = []
    labels, short = [], []
    for record in records:
        cells = {}
        for row in record["runs"]:
            if row["measured"]:
                key = (row["phase"], row["length"], row["mtp_tokens"])
                cells.setdefault(key, []).append(row["prefill_tok_per_s"] if row["phase"] == "C" else row["decode_tok_per_s"])
        data.append(cells)
        env = record["environment"]
        state = "/".join(env["recurrent_state"]).replace("torch.float", "FP")
        fuzev = env["engine"].startswith("exllamav3-fuzev")
        short.append(f"{record['label']}, {state} state" + ("" if fuzev or state == "FP32" else " (default)"))
        labels.append(f"{record['label']} ({env['engine_build']}), {state} state" if fuzev else short[-1])
    lengths = sorted(key[1] for key in data[1] if key[0] == "C")
    # Compact log2 positions omit unmeasured steps instead of reserving an empty slot.
    x = np.arange(len(lengths))
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), gridspec_kw={"width_ratios": [1.5, 1]})
    for cells, (_, color, style, marker), label in zip(data, SPEED_ARMS, labels):
        values = [cells["C", length, 0] for length in lengths]
        mean = np.array([np.mean(value) for value in values])
        axes[0].errorbar(x, mean, yerr=[mean - [min(v) for v in values], [max(v) for v in values] - mean],
                         color=color, ls=style, marker=marker, capsize=3, label=label)
    axes[0].set_title("Prefill Throughput")
    axes[0].set_ylabel("tokens/s (mean; whiskers = min-max repetitions)")
    for other in (0, 2, 3):
        a = [data[1]["C", n, 0] for n in lengths]
        b = [data[other]["C", n, 0] for n in lengths]
        mean = np.array([np.mean(v) / np.mean(w) for v, w in zip(a, b)])
        low = np.array([min(v) / max(w) for v, w in zip(a, b)])
        high = np.array([max(v) / min(w) for v, w in zip(a, b)])
        _, color, style, marker = SPEED_ARMS[other]
        axes[1].errorbar(x, mean, yerr=[mean - low, high - mean], color=color, ls=style,
                         marker=marker, capsize=3, label=f"FuZev v1.1 / {short[other]}")
        for position, value in zip(x, mean):
            axes[1].annotate(f"{value:.2f}x", (position, value), xytext=(0, 8 if other != 3 else -13),
                             textcoords="offset points", ha="center", fontsize=7)
    axes[1].axhline(1, color="#222222", lw=0.8)
    axes[1].set_title("FuZev v1.1 Prefill Ratio")
    axes[1].legend(fontsize=6.5, frameon=False)
    for ax in axes:
        ax.set_xticks(x, [f"{n // 1024}k" for n in lengths])
        ax.set_xlabel("prompt tokens (compact log2 steps; 16k skipped)")
    fig.suptitle("Prefill on the Same 4.05 bpw Backbone", fontweight="bold")
    save(fig, out, "prefill", speed_notes(records, "C") +
         ["Ratio whiskers: lowest v1.1 repetition / highest reference through highest v1.1 / lowest reference."],
         groups=speed_groups(*axes[0].get_legend_handles_labels()))

    mtps = sorted({key[2] for key in data[1] if key[0] == "D"})
    fig, axes = plt.subplots(1, len(mtps), figsize=(13, 10), squeeze=False)
    width = 0.12
    for ax, mtp in zip(axes[0], mtps):
        lengths = sorted(key[1] for key in data[1] if key[0] == "D" and key[2] == mtp)
        x = np.arange(len(lengths))
        all_values = []
        for index, (cells, (_, color, style, _), label) in enumerate(zip(data, SPEED_ARMS, labels)):
            values = [cells["D", length, mtp] for length in lengths]
            means = np.array([np.mean(v) for v in values])
            ax.bar(x + (index - 2.5) * width, means, width, label=label,
                color="white" if style == "--" else color, edgecolor=color,
                hatch="///" if style == "--" else None,
                yerr=[means - [min(v) for v in values], [max(v) for v in values] - means], capsize=2)
            all_values.extend(value for values_ in values for value in values_)
        if mtp == 0:
            span = max(all_values) - min(all_values)
            ax.set_ylim(min(all_values) - span * 0.18, max(all_values) + span * 0.18)
        else:
            ax.set_ylim(0, max(all_values) * 1.12)
        ax.set_xticks(x, [f"{n // 1024}k" for n in lengths])
        ax.set_xlabel("prompt tokens (compact log2 steps; 16k skipped)")
        ax.set_ylabel("decode tokens/s (mean; whiskers = min-max)")
        outputs = next(row["expected_completion_tokens"] for row in records[1]["runs"] if row["phase"] == "D")
        ax.set_title(f"{outputs:,} output tokens; " + ("MTP off" if mtp == 0 else f"MTP {mtp}"))
    fig.suptitle("Decode Throughput", fontweight="bold")
    save(fig, out, "decode", speed_notes(records, "D"), groups=speed_groups(*axes[0][0].get_legend_handles_labels()))


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--results", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.2, "axes.axisbelow": True, "svg.fonttype": "none"})
    reference = read(args.results, "kev-published.json")
    index_chart(args.results, args.out, reference)
    area_chart(args.results, args.out, reference)
    delta_chart(args.results, args.out)
    image_chart(args.results, args.out)
    speed_charts(args.results, args.out)


if __name__ == "__main__":
    main()
