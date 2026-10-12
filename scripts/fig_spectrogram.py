"""The report's "from sound to prediction" figure: one POW clip, its call boxes, and how each model slices it.

    python -I scripts/fig_spectrogram.py [--choose] [--clip configs/reference/figure_clip.yaml]
                                         [--out docs/assets] [--no-audio]

Source: BirdSet's POW soundscapes (CC0). --choose picks the clip by a fixed rule
and records it in --clip: among POW's 5 s segments with a full 5 s neighbour on
each side, the one with two or three labelled species (calls overlapping the
segment by >= 0.5 s, the harness's rule) and the most annotated calls; ties go
to the earliest file and time. Without --choose the recorded clip is redrawn.

Writes to --out:
  spectrogram-light.png, spectrogram-dark.png   mel spectrogram, 0-16 kHz, at 2x
  fig_spectrogram.svg   overlay drawn over the PNGs (inlined by render_report.py):
                        call boxes, the labelled segment and each model's windows;
                        a window is filled where it counts for the segment under
                        the half-window rule (more than half of it inside)
  og.png                1200 x 630 social preview from the same figure
  clip.mp3              the 15 s excerpt (unless --no-audio)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy.signal import resample_poly, stft

SR = 32_000
FMAX = 16_000
N_MELS = 160
CONTEXT_S = 5.0  # neighbouring segment shown on each side
SEG_S = 5.0
MODELS = [  # (label, colour slot, window s, hop s), as configured in birdeval
    ("BirdNET 2.4", "m1", 3.0, 1.5),
    ("BirdNET 3.0", "m2", 3.0, 1.5),
    ("Perch V2", "m3", 5.0, 2.5),
]
POW_META = "POW/POW_metadata_test.parquet"
HF_SNAPSHOT = Path.home() / ".cache/huggingface/hub/datasets--DBD-research-group--BirdSet/snapshots"
HF_EXTRACTED = Path.home() / ".cache/huggingface/datasets/downloads/extracted"
EBIRD = Path("configs/taxonomy/ebird_taxonomy.csv")
# colours of the light and dark spectrograms: page surface -> page ink (render_report.py tokens)
INK = {"light": ("#ffffff", "#17201d"), "dark": ("#18211e", "#e4ebe8")}


@dataclass
class Box:
    start: float  # s, excerpt time
    end: float
    low: float  # Hz
    high: float
    label: str


# ---- mel spectrogram -----------------------------------------------------------------

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m, float) / 2595.0) - 1.0)


def mel_spectrogram(wave: np.ndarray, sr: int = SR, n_mels: int = N_MELS, fmax: float = FMAX) -> np.ndarray:
    """dB mel power spectrogram [n_mels, frames], top 80 dB, numpy/scipy only."""
    nper, hop = 1024, 160
    f, _, z = stft(wave, fs=sr, nperseg=nper, noverlap=nper - hop, boundary=None, padded=False)
    power = np.abs(z) ** 2
    edges = mel_to_hz(np.linspace(0, hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (f - lo) / max(mid - lo, 1e-9)
        down = (hi - f) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(up, down), 0, None)
    mel = fb @ power
    db = 10 * np.log10(np.maximum(mel, 1e-12))
    return np.maximum(db, db.max() - 80.0)


def save_png(spec: np.ndarray, path: Path, colors: tuple[str, str], size_px: tuple[int, int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("ink", list(colors))
    w, h = size_px
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(spec, origin="lower", aspect="auto", cmap=cmap, interpolation="bilinear")
    ax.set_axis_off()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    quantize(path)


def quantize(path: Path, colors: int = 16) -> None:
    """A two-colour ramp needs few levels: a palette PNG is a fraction of the size."""
    from PIL import Image

    with Image.open(path) as im:
        q = im.convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q.save(path, optimize=True)


# ---- the overlay ----------------------------------------------------------------------

def counts_for_segment(w0: float, w1: float, s0: float, s1: float) -> bool:
    """Half-window rule: a window counts for a segment when more than half of it lies inside."""
    return max(0.0, min(w1, s1) - max(w0, s0)) > (w1 - w0) / 2


def windows(hop: float, length: float, file_offset: float, span: float) -> list[tuple[float, float]]:
    """Windows in excerpt time: they start every hop from the file's start, as the models slice it."""
    first = -((file_offset % hop) or 0.0)
    out, t = [], first
    while t < span:
        if t + length > 0:
            out.append((t, t + length))
        t += hop
    return out


def overlay_svg(boxes: list[Box], span: float, seg: tuple[float, float], file_offset: float,
                img: dict[str, str]) -> str:
    """Inline SVG: spectrogram images, axes, call boxes, the segment and the model windows."""
    W, left, right, top = 900, 64, 16, 34
    pw, ph = W - left - right, 250
    lane_h, lane_gap = 9, 3
    rows_top = top + ph + 40
    row_h = 2 * lane_h + lane_gap + 14
    H = rows_top + row_h * len(MODELS) + 6
    x = lambda t: left + t / span * pw  # noqa: E731
    y = lambda hz: top + ph - float(hz_to_mel(hz) / hz_to_mel(FMAX)) * ph  # noqa: E731
    o = [f'<svg class="fig-spec" viewBox="0 0 {W} {H}" role="img" aria-label="Mel spectrogram of a 15-second '
         f'POW recording excerpt with BirdSet call boxes, the labelled 5-second segment, and the windows each '
         f'model scores">']
    for theme, href in img.items():
        o.append(f'<image class="spec-{theme}" href="{escape(href)}" x="{left}" y="{top}" width="{pw}" height="{ph}" '
                 f'preserveAspectRatio="none"/>')
    o.append(f'<rect class="spec-frame" x="{left}" y="{top}" width="{pw}" height="{ph}"/>')
    for khz in [1, 2, 4, 8, 16]:
        yy = y(khz * 1000)
        o.append(f'<line class="gridline" x1="{left - 4}" x2="{left}" y1="{yy:.1f}" y2="{yy:.1f}"/>'
                 f'<text class="tick" x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end">{khz}</text>')
    o.append(f'<text class="tick" x="{left - 8}" y="{top - 10}" text-anchor="end">kHz</text>')
    for t in np.arange(0, span + 1e-9, 2.5):
        o.append(f'<text class="tick" x="{x(t):.1f}" y="{top + ph + 16}" text-anchor="middle">{t:g}</text>')
    o.append(f'<text class="tick" x="{left + pw}" y="{top + ph + 30}" text-anchor="end">seconds</text>')
    # the labelled segment and its neighbours
    s0, s1 = seg
    o.append(f'<rect class="seg-hi" x="{x(s0):.1f}" y="{top}" width="{x(s1) - x(s0):.1f}" height="{ph}"/>')
    for t in (s0, s1):
        o.append(f'<line class="seg-edge" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{rows_top + row_h * len(MODELS)}"/>')
    o.append(f'<text class="glabel" x="{(x(s0) + x(s1)) / 2:.1f}" y="{top - 12}" text-anchor="middle">labelled 5-second segment</text>')
    o.append(f'<text class="gsub" x="{(left + x(s0)) / 2:.1f}" y="{top - 12}" text-anchor="middle">previous segment</text>')
    o.append(f'<text class="gsub" x="{(x(s1) + left + pw) / 2:.1f}" y="{top - 12}" text-anchor="middle">next segment</text>')
    # call boxes; each species named once, at the box overlapping the labelled segment most
    def score(b):
        return (max(0.0, min(b.end, s1) - max(b.start, s0)), (b.end - b.start) * (b.high - b.low))
    anchor_box = {}
    for b in boxes:
        if b.label not in anchor_box or score(b) > score(anchor_box[b.label]):
            anchor_box[b.label] = b
    for b in sorted(boxes, key=lambda b: b.start):
        b0, b1 = max(b.start, 0), min(b.end, span)
        if b1 <= b0:
            continue
        yy0, yy1 = y(b.high), y(b.low)
        o.append(f'<rect class="callbox" x="{x(b0):.1f}" y="{yy0:.1f}" width="{x(b1) - x(b0):.1f}" height="{yy1 - yy0:.1f}">'
                 f'<title>{escape(b.label)}: {b.start:.1f}–{b.end:.1f} s, {b.low / 1000:.1f}–{b.high / 1000:.1f} kHz</title></rect>')
    placed: list[tuple[float, float, float, float]] = []
    for label, b in sorted(anchor_box.items(), key=lambda kv: -kv[1].high):
        w = 7.0 * len(label) + 6
        lx = min(max(x(max(b.start, 0)), left + 4), left + pw - w - 4)
        ly = max(y(b.high) - 6, top + 14)
        clash = lambda yy: any(lx < r[2] and lx + w > r[0] and yy - 12 < r[3] and yy + 2 > r[1] for r in placed)  # noqa: E731
        while clash(ly) and ly < top + ph - 4:
            ly += 15
        placed.append((lx, ly - 12, lx + w, ly + 2))
        o.append(f'<text class="boxlabel" x="{lx:.1f}" y="{ly:.1f}">{escape(label)}</text>')
    # model windows, in two staggered lanes so overlapping windows stay visible
    for i, (name, cls, length, hop) in enumerate(MODELS):
        ry = rows_top + i * row_h
        o.append(f'<text class="gsub" x="{left - 8}" y="{ry + lane_h + 6}" text-anchor="end">{escape(name.split()[0])}</text>'
                 f'<text class="gsub" x="{left - 8}" y="{ry + lane_h + 18}" text-anchor="end">{escape(" ".join(name.split()[1:]))}</text>')
        for j, (w0, w1) in enumerate(windows(hop, length, file_offset, span)):
            c0, c1 = max(w0, 0), min(w1, span)
            on = counts_for_segment(w0, w1, s0, s1)
            yy = ry + (j % 2) * (lane_h + lane_gap)
            tip = f"{name} window {w0:.1f}–{w1:.1f} s: " + ("counts for the labelled segment" if on else "does not count for it")
            o.append(f'<g class="{cls}"><title>{escape(tip)}</title><rect class="win{" on" if on else ""}" '
                     f'x="{x(c0) + 1:.1f}" y="{yy}" width="{max(x(c1) - x(c0) - 2, 1):.1f}" height="{lane_h}" rx="3"/></g>')
        o.append(f'<text class="gsub" x="{left + pw}" y="{ry + 2 * lane_h + lane_gap + 12}" text-anchor="end">'
                 f'{length:g} s windows every {hop:g} s</text>')
    o.append("</svg>")
    return "".join(o)


def og_png(spec: np.ndarray, boxes: list[Box], span: float, seg: tuple[float, float], path: Path, title: str) -> None:
    """1200 x 630 preview: the light spectrogram with the segment, boxes and a title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor="#f5f7f6")
    fig.text(0.05, 0.9, title, fontsize=26, fontweight="bold", color="#17201d", va="center")
    fig.text(0.05, 0.83, "BirdNET 2.4 · BirdNET 3.0 · Perch V2 on the BirdSet test sites", fontsize=15,
             color="#56635e", va="center")
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.68])
    ax.imshow(spec, origin="lower", aspect="auto", extent=[0, span, 0, 1],
              cmap=LinearSegmentedColormap.from_list("ink", list(INK["light"])))
    ax.add_patch(Rectangle((seg[0], 0), seg[1] - seg[0], 1, fill=False, ec="#1d5c4b", lw=3))
    top = float(hz_to_mel(FMAX))
    for b in boxes:
        y0, y1 = float(hz_to_mel(b.low)) / top, float(hz_to_mel(b.high)) / top
        ax.add_patch(Rectangle((b.start, y0), b.end - b.start, y1 - y0, fill=False, ec="#c26a00", lw=2))
    ax.set_xlim(0, span)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    quantize(path, 64)


def make_figure(wave: np.ndarray, sr: int, boxes: list[Box], seg: tuple[float, float], out: Path,
                file_offset: float = 0.0, audio: bool = True, title: str = "Bird-sound models on BirdSet") -> list[Path]:
    """Draw everything from one excerpt; returns the files written."""
    out.mkdir(parents=True, exist_ok=True)
    if sr != SR:
        wave = resample_poly(wave, SR, sr)
        sr = SR
    span = len(wave) / sr
    spec = mel_spectrogram(wave, sr)
    written = []
    for theme, colors in INK.items():
        p = out / f"spectrogram-{theme}.png"
        save_png(spec, p, colors, (1640, 500))
        written.append(p)
    svg = overlay_svg(boxes, span, seg, file_offset,
                      {t: f"assets/spectrogram-{t}.png" for t in INK})
    (out / "fig_spectrogram.svg").write_text(svg)
    written.append(out / "fig_spectrogram.svg")
    og_png(spec, boxes, span, seg, out / "og.png", title)
    written.append(out / "og.png")
    if audio:
        peak = np.max(np.abs(wave)) or 1.0
        sf.write(out / "clip.mp3", (wave / peak * 0.9).astype(np.float32), sr, format="MP3")
        written.append(out / "clip.mp3")
    return written


# ---- choosing and loading the POW clip ------------------------------------------------------

def pow_events() -> pd.DataFrame:
    snaps = sorted(HF_SNAPSHOT.glob(f"*/{POW_META}"))
    if not snaps:
        raise SystemExit(f"no {POW_META} under {HF_SNAPSHOT}; run a BirdSet POW evaluation first")
    df = pd.read_parquet(snaps[-1])
    if df.index.name == "filepath":
        df = df.reset_index()
    df["file"] = df["filepath"].map(lambda p: Path(str(p)).name)
    return df.dropna(subset=["low_freq", "high_freq"])


def audio_path(name: str) -> Path:
    hits = sorted(HF_EXTRACTED.glob(f"POW_test_shard_*/{name}"))
    if not hits:
        raise SystemExit(f"{name} not found under {HF_EXTRACTED}/POW_test_shard_*")
    return hits[0]


def choose(ev: pd.DataFrame) -> dict:
    best = None
    for name, g in sorted(ev.groupby("file"), key=lambda kv: kv[0]):
        dur = sf.info(str(audio_path(name))).duration
        for s in np.arange(CONTEXT_S, dur - SEG_S - CONTEXT_S + 1e-9, SEG_S):
            ov = np.minimum(g["end_time"], s + SEG_S) - np.maximum(g["start_time"], s)
            inside = g[ov >= 0.5]
            n_sp = inside["ebird_code"].nunique()
            if 2 <= n_sp <= 3:
                key = (-len(inside), name, s)
                if best is None or key < best[0]:
                    best = (key, {"file": name, "segment_start_s": float(s),
                                  "species": sorted(inside["ebird_code"].unique().tolist())})
    if best is None:
        raise SystemExit("no POW segment has two or three labelled species")
    return best[1]


def common_names() -> dict[str, str]:
    if not EBIRD.exists():
        return {}
    eb = pd.read_csv(EBIRD)
    cols = {c.upper(): c for c in eb.columns}
    return dict(zip(eb[cols["SPECIES_CODE"]], eb[cols.get("PRIMARY_COM_NAME") or cols["COMMON_NAME"]]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, default=Path("configs/reference/figure_clip.yaml"))
    ap.add_argument("--choose", action="store_true", help="pick the clip by the rule and record it in --clip")
    ap.add_argument("--out", type=Path, default=Path("docs/assets"))
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    ev = pow_events()
    if args.choose or not args.clip.exists():
        clip = choose(ev)
        clip |= {"source": "BirdSet POW soundscapes (Powdermill Nature Reserve), CC0 1.0",
                 "rule": "two or three labelled species in the segment, most annotated calls; earliest on ties",
                 "context_s": CONTEXT_S}
        args.clip.write_text("# Written by scripts/fig_spectrogram.py --choose\n" + yaml.safe_dump(clip, sort_keys=False))
        print(f"chose {clip['file']} at {clip['segment_start_s']:g} s: {', '.join(clip['species'])}")
    clip = yaml.safe_load(args.clip.read_text())
    s = clip["segment_start_s"]
    t0 = s - CONTEXT_S
    path = audio_path(clip["file"])
    info = sf.info(str(path))
    wave, sr = sf.read(str(path), start=int(t0 * info.samplerate), stop=int((s + SEG_S + CONTEXT_S) * info.samplerate),
                       dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    names = common_names()
    g = ev[ev["file"] == clip["file"]]
    g = g[(g["end_time"] > t0) & (g["start_time"] < s + SEG_S + CONTEXT_S)]
    boxes = [Box(r.start_time - t0, r.end_time - t0, r.low_freq, r.high_freq, names.get(r.ebird_code, r.ebird_code))
             for r in g.itertuples()]
    for p in make_figure(wave, sr, boxes, (CONTEXT_S, CONTEXT_S + SEG_S), args.out, file_offset=t0,
                         audio=not args.no_audio):
        print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
