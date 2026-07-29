"""Crop the camera to follow the instructor, keeping him centred.

Produces <input>_tracked.mp4: a zoomed 16:9 crop of the camera that pans to keep
the instructor in the middle of frame. Intended as the picture-in-picture source
for src/assembly/assembly.py -- at 480x270 the instructor is a few dozen pixels
tall in the raw wide shot, and a 2x crop makes him actually visible.

Run this AFTER face_anon, so the crop is taken from the anonymized camera and
student faces stay pixelated. A tight crop on the instructor also tends to
exclude the audience from frame altogether, which helps rather than hurts.

Why the pan looks stable
------------------------
This is offline, so the smoothing filter can be *centred* (non-causal): each
frame's crop position is computed from detections both before and after it. A
live autotracking camera can only use the past, so it always trails the subject;
here there is no phase lag at all. Measured on lecture 12, the instructor's face
moves a median 0.67 px/frame, so a 2-second Hann window removes detection jitter
without visibly lagging real movement.

Three further guards against a nauseating result:
  * detection gaps hold the last known position rather than snapping to centre;
  * an optional dead band (--deadzone) lets him drift a few px before the crop
    moves at all, then slides along the band edge instead of jumping;
  * the crop is clamped inside the frame, so it stops panning at the edges
    rather than exposing black bars.

Usage:
    python -m src.video.track_instructor --lecture-dir data/15210-lecture12
    python -m src.video.track_instructor --lecture-dir data/... --zoom 0.45
    python -m src.video.track_instructor --lecture-dir data/... --preview
"""

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

from src.video.face_anon import (build_app, cpu_count, detect_boxes, embed_face,
                                 identify_instructor, probe)


# ---------------------------------------------------------------------------
# Pass 1: where is the instructor, frame by frame
# ---------------------------------------------------------------------------
def locate_instructor(video_path, app, prototypes, start_frame, n_frames,
                      detect_every=8, sim_threshold=0.45, label=""):
    """Return (frames, cx, cy) arrays of sampled instructor face centres.

    Only the instructor is tracked: every detected face is matched against the
    identity prototypes and non-instructor faces are ignored, so a student
    walking through frame does not drag the camera with them.
    """
    cap = cv2.VideoCapture(video_path)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    protos = np.atleast_2d(prototypes) if prototypes is not None else None
    fr, xs, ys, sizes = [], [], [], []
    n_infer = 0
    t0 = time.time()
    for off in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            n_frames = off
            break
        if off % detect_every:
            continue
        boxes, kpss = detect_boxes(app, frame)
        n_infer += 1
        best = None
        for j in range(len(boxes)):
            box = boxes[j][:4]
            if protos is None:
                sim = 1.0
            else:
                emb = embed_face(app, frame, box, kpss[j], boxes[j][4])
                if emb is None:
                    continue
                sim = float(np.max(protos @ emb))
            if sim < sim_threshold:
                continue
            # If several faces match, take the largest -- the instructor is the
            # nearest matching face to camera.
            area = (box[2] - box[0]) * (box[3] - box[1])
            if best is None or area > best[2]:
                best = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, area)
        if best is not None:
            fr.append(off); xs.append(best[0]); ys.append(best[1])
            sizes.append(float(np.sqrt(best[2])))
        if n_infer % 500 == 0:
            el = time.time() - t0
            print(f"[track] {label}pass1 {off}/{n_frames} "
                  f"({100.0 * off / max(n_frames, 1):.1f}%), {len(fr)} hits "
                  f"@ {n_infer / max(el, 1e-9):.1f} det/s", flush=True)
    cap.release()
    print(f"[track] {label}pass1 done: {n_infer} detections, {len(fr)} instructor "
          f"hits in {(time.time() - t0) / 60:.1f} min", flush=True)
    return (np.array(fr), np.array(xs), np.array(ys), np.array(sizes), n_frames)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def _hann_smooth(v, win):
    """Centred Hann-weighted moving average. Edges use a shrinking window so the
    first and last frames are not pulled toward zero."""
    win = max(1, int(win) | 1)                 # odd, so it is symmetric
    if win <= 1 or len(v) < 3:
        return v.astype(float)
    w = np.hanning(win + 2)[1:-1]
    w /= w.sum()
    pad = win // 2
    vp = np.pad(v.astype(float), pad, mode="edge")
    return np.convolve(vp, w, mode="valid")


def _dead_band(v, deadzone):
    """Hold position until the target drifts past `deadzone`, then slide along
    the band edge. Sliding rather than snapping avoids visible jumps."""
    if deadzone <= 0:
        return v
    out = np.empty_like(v)
    cur = v[0]
    for i, target in enumerate(v):
        d = target - cur
        if abs(d) > deadzone:
            cur = target - np.sign(d) * deadzone
        out[i] = cur
    return out


def build_crop_path(fr, xs, ys, n_frames, fps, meta, zoom=0.5,
                    smooth_seconds=2.0, deadzone=0.0):
    """Per-frame crop rectangle (x, y, w, h), clamped inside the frame."""
    W, H = meta["width"], meta["height"]
    cw = int(round(np.clip(zoom, 0.15, 1.0) * W))
    ch = int(round(cw * H / W))                # keep the source aspect ratio
    cw, ch = min(cw, W), min(ch, H)

    all_f = np.arange(n_frames)
    if len(fr) == 0:
        # Never found him: centre the crop and don't pretend to track.
        print("[track] WARNING no instructor detections; using a static "
              "centre crop", flush=True)
        cx = np.full(n_frames, W / 2.0)
        cy = np.full(n_frames, H / 2.0)
    else:
        # np.interp holds the first/last value outside the sampled range, which
        # is exactly the "hold last known position" behaviour we want for gaps.
        cx = np.interp(all_f, fr, xs)
        cy = np.interp(all_f, fr, ys)

    win = max(1, int(round(smooth_seconds * fps)))
    cx = _dead_band(_hann_smooth(cx, win), deadzone)
    cy = _dead_band(_hann_smooth(cy, win), deadzone)

    # Frame the subject slightly above centre: a head centred in the box puts
    # the body out of shot, which reads as badly composed.
    cy = cy + ch * 0.10

    x = np.clip(np.round(cx - cw / 2.0), 0, W - cw).astype(int)
    y = np.clip(np.round(cy - ch / 2.0), 0, H - ch).astype(int)
    return x, y, cw, ch


# ---------------------------------------------------------------------------
# Pass 2: crop + encode
# ---------------------------------------------------------------------------
def render(video_path, out_path, x, y, cw, ch, out_w, out_h, fps,
           start_frame=0, crf=20, threads=None, has_audio=True):
    threads = threads or max(1, cpu_count())
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}",
        "-r", f"{fps:.10g}", "-i", "pipe:0",
    ]
    if has_audio:
        cmd += ["-i", video_path, "-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-b:a", "128k", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-threads", str(threads),
            "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(video_path)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    t0 = time.time()
    try:
        for i in range(len(x)):
            ok, frame = cap.read()
            if not ok:
                break
            crop = frame[y[i]:y[i] + ch, x[i]:x[i] + cw]
            if crop.shape[0] != out_h or crop.shape[1] != out_w:
                crop = cv2.resize(crop, (out_w, out_h),
                                  interpolation=cv2.INTER_LANCZOS4)
            proc.stdin.write(np.ascontiguousarray(crop).tobytes())
            written += 1
            if written % 2000 == 0:
                print(f"[track] pass2 {written}/{len(x)} "
                      f"({100.0 * written / len(x):.1f}%) @ "
                      f"{written / max(time.time() - t0, 1e-9):.0f} fps", flush=True)
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()
    return written


def save_preview(video_path, x, y, cw, ch, out_png, n=6, start_frame=0):
    """Contact sheet of sampled frames with the crop rectangle drawn on, so the
    framing can be checked before committing to a full encode."""
    cap = cv2.VideoCapture(video_path)
    rows = []
    for i in np.linspace(0, len(x) - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + int(i))
        ok, f = cap.read()
        if not ok:
            continue
        cv2.rectangle(f, (x[i], y[i]), (x[i] + cw, y[i] + ch), (0, 255, 0), 3)
        cv2.putText(f, f"frame {i}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        rows.append(cv2.resize(f, (640, 360)))
    cap.release()
    if not rows:
        return None
    grid = np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows) - 1, 2)])
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    cv2.imwrite(out_png, grid)
    return out_png


def _has_audio(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout
    return "audio" in out


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lecture-dir", required=True)
    p.add_argument("--input", default=None,
                   help="Defaults to camera_muted_anon.mp4 (run after face_anon "
                        "so the crop keeps student faces pixelated), else "
                        "camera_muted.mp4, else camera.mp4")
    p.add_argument("--out", default=None, help="Default: <input>_tracked.mp4")
    p.add_argument("--zoom", type=float, default=0.5,
                   help="Crop width as a fraction of frame width. 0.5 = 2x zoom "
                        "(default); lower is tighter")
    p.add_argument("--smooth-seconds", type=float, default=2.0,
                   help="Centred smoothing window. Larger = steadier but slower "
                        "to follow real movement")
    p.add_argument("--deadzone", type=float, default=0.0,
                   help="Pixels the instructor may drift before the crop moves "
                        "at all (0 = always centre him)")
    p.add_argument("--out-size", default=None,
                   help="WxH of the output, e.g. 960x540. Default: the crop size")
    p.add_argument("--detect-every", type=int, default=8)
    p.add_argument("--det-size", type=int, default=640)
    p.add_argument("--sample-frames", type=int, default=400)
    p.add_argument("--sim-threshold", type=float, default=0.45)
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--preview", action="store_true",
                   help="Write track_preview.png with the crop box drawn, exit")
    p.add_argument("--save-path", default=None,
                   help="Dump raw detections + crop path as JSON, so a later "
                        "--zoom change can reuse them via --from-path")
    p.add_argument("--from-path", default=None,
                   help="Reuse detections from a --save-path JSON instead of "
                        "re-detecting. Changing --zoom or --smooth-seconds then "
                        "costs only the render pass")
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Prefer the anonymized camera so the crop cannot un-blur a student.
    if args.input:
        name = args.input
    else:
        for cand in ("camera_muted_anon.mp4", "camera_muted.mp4", "camera.mp4"):
            if os.path.exists(os.path.join(args.lecture_dir, cand)):
                name = cand
                break
        else:
            raise FileNotFoundError(f"no camera video in {args.lecture_dir}")
    video = os.path.join(args.lecture_dir, name)
    if "anon" not in name:
        print(f"[track] WARNING tracking {name}, which is NOT face-anonymized. "
              f"Zooming an un-anonymized camera makes any student in frame MORE "
              f"identifiable. Run face_anon first.", flush=True)

    base = os.path.splitext(name)[0]
    out_path = args.out or os.path.join(args.lecture_dir, f"{base}_tracked.mp4")
    meta = probe(video)
    fps = meta["fps"]
    n_frames = meta["n_frames"]
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)
    print(f"[track] {video}: {meta['width']}x{meta['height']} @ {fps:g}fps, "
          f"{n_frames} frames")

    if args.from_path:
        with open(args.from_path) as f:
            saved = json.load(f)
        d = saved["detections"]
        fr = np.array(d["frames"]); xs = np.array(d["cx"]); ys = np.array(d["cy"])
        n_frames = min(n_frames, int(saved["n_frames"]))
        print(f"[track] reusing {len(fr)} saved detections from "
              f"{args.from_path} -- skipping the detection pass")
    else:
        app = build_app(det_size=args.det_size, need_recognition=True)
        protos, clusters, info = identify_instructor(
            video, app, sample_frames=args.sample_frames,
            start_frame=0, end_frame=meta["n_frames"])
        print(f"[track] identity: {info}")

        fr, xs, ys, sizes, n_frames = locate_instructor(
            video, app, protos, 0, n_frames,
            detect_every=args.detect_every, sim_threshold=args.sim_threshold)
    hit_rate = 100.0 * len(fr) / max(1, n_frames / args.detect_every)
    print(f"[track] instructor located in {hit_rate:.1f}% of sampled frames"
          + ("" if hit_rate > 40 else "  <-- low; check --sim-threshold"))

    x, y, cw, ch = build_crop_path(fr, xs, ys, n_frames, fps, meta,
                                   zoom=args.zoom,
                                   smooth_seconds=args.smooth_seconds,
                                   deadzone=args.deadzone)
    pan = float(np.abs(np.diff(x)).mean()) if len(x) > 1 else 0.0
    print(f"[track] crop {cw}x{ch} ({args.zoom:g} of width -> "
          f"{meta['width'] / cw:.2f}x zoom), mean pan {pan:.2f} px/frame")

    if args.save_path:
        # Save the RAW detections, not just the derived crop. Detection is the
        # entire cost of this stage (16 min for a 79-minute lecture); the crop is
        # cheap arithmetic on top. Keeping the detections means changing --zoom or
        # --smooth-seconds later is a re-render, not a re-detect.
        with open(args.save_path, "w") as f:
            json.dump({"detections": {"frames": fr.tolist(), "cx": xs.tolist(),
                                      "cy": ys.tolist()},
                       "n_frames": int(n_frames), "fps": fps,
                       "width": meta["width"], "height": meta["height"],
                       "crop": {"w": cw, "h": ch,
                                "x": x.tolist(), "y": y.tolist()}}, f)
        print(f"[track] detections + crop path -> {args.save_path}")

    if args.preview:
        png = save_preview(video, x, y, cw, ch,
                           os.path.join(args.lecture_dir, "track_preview.png"))
        print(f"[track] preview: {png}\n"
              f"  Green box is the crop. Re-run with --zoom / --smooth-seconds "
              f"to adjust, then drop --preview.")
        return

    if args.out_size:
        ow, oh = (int(v) for v in args.out_size.lower().split("x"))
    else:
        ow, oh = cw, ch
    written = render(video, out_path, x, y, cw, ch, ow, oh, fps,
                     crf=args.crf, threads=args.threads,
                     has_audio=_has_audio(video))
    size = os.path.getsize(out_path) / 1e6
    print(f"[track] wrote {out_path} ({written} frames, {ow}x{oh}, {size:.1f} MB)")


if __name__ == "__main__":
    main()
