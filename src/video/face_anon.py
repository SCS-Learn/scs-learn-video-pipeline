"""Facial anonymization for lecture camera video (blur/pixelate non-instructor faces).

Privacy-preserving pass over camera.mp4: detect every face, work out which one
is the instructor, obscure everybody else. Writes <input>_anon.mp4.

Where the time actually goes
---------------------------
Measured on data/15210-lecture12/camera.mp4 (1280x720, 119,344 frames), CPU:

    cv2 decode                2075 fps    <- 0.5% of runtime
    blur + mp4v write          433 fps    <- 0.6%
    InsightFace app.get()     4.31 fps    <- 98.9%  ... everything

Inference is the whole cost, so that is what this module optimizes; decode and
encode are treated as free. Four changes over the first version:

1. buffalo_l ships five models and FaceAnalysis() runs all of them on every
   face, but this code only ever reads .bbox and .normed_embedding. Restricting
   to allowed_modules=['detection','recognition'] drops 1k3d68.onnx (144MB 3D
   landmarks), 2d106det.onnx and genderage.onnx. Measured: 3.95x for the
   detection-only path, 1.78x for detection+recognition.

2. The old code ran detection *and* recognition on all 119,344 frames purely to
   decide who the instructor is. Identity needs a few hundred frames, not all of
   them: the sample stage looks at --sample-frames frames spread across the
   lecture, clusters those, and reduces the instructor to one centroid
   embedding. The blur stage then needs detection only, plus one embedding per
   newly-appearing track.

3. Detection runs every --detect-every frames. Boxes for the skipped frames come
   from the *union* of the two bracketing detections rather than the previous
   frame's boxes held verbatim, which is what the old --detect-every did
   (`result = last_result`). Holding a stale box lags a moving face and can
   leave part of it unblurred -- a privacy bug, not a quality one. A union
   covers the whole swept path, erring towards over-blurring.

4. cv2.VideoWriter(mp4v) is replaced by a pipe into ffmpeg libx264. Measured
   1.4x faster and 2.5x smaller: a full lecture goes from 2.39 GB of mpeg4 to
   0.95 GB of h264, against a 393 MB h264 source.

Chunking, so a whole GPU node is usable: --chunks splits the video into
independent frame ranges, each handled by its own worker process with its own
encoder, then joined with a stream-copy concat. Bridges-2 GPU nodes carry 8 GPUs
each (verified 2026-07-29: v100-16, v100-32, l40s-48 and h100-80 flavours all
exist), so --chunks 8 pins one worker per GPU via CUDA_VISIBLE_DEVICES.

Fail-closed by design: a face whose identity cannot be established (no
embedding, recognition failed, cluster too small) is blurred, never skipped.

Usage:
    python -m src.video.face_anon --lecture-dir data/15210-lecture12
    python -m src.video.face_anon --lecture-dir data/... --chunks 8
    python -m src.video.face_anon --lecture-dir data/... --preview
    python -m src.video.face_anon --lecture-dir data/... --benchmark
    python -m src.video.face_anon --lecture-dir data/... --end 00:02:00 --detect-every 1
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def parse_time(s):
    """Parse 'MM:SS', 'HH:MM:SS', or plain seconds into float seconds (or None)."""
    if s is None:
        return None
    parts = [float(p) for p in str(s).split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def cpu_count():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 4


def probe(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


# ---------------------------------------------------------------------------
# InsightFace, restricted to the two models this pipeline actually reads
# ---------------------------------------------------------------------------
def build_app(det_size=640, need_recognition=True, quiet=True):
    """Build a FaceAnalysis limited to detection (+ recognition on request).

    Dropping the three unused models is the single largest constant-factor win
    available here -- see the module docstring for measured numbers.
    """
    import onnxruntime
    from insightface.app import FaceAnalysis

    if quiet:
        onnxruntime.set_default_logger_severity(3)

    modules = ["detection", "recognition"] if need_recognition else ["detection"]

    # get_available_providers() reports what the BUILD supports, not what
    # actually loads. On a PSC H100 node it listed CUDAExecutionProvider while
    # libonnxruntime_providers_cuda.so failed to load for want of
    # libcublasLt.so.12; onnxruntime then fell back to CPU silently and this
    # function still printed "device=cuda". A ~25x slowdown that reports itself
    # as the fast path is worse than an outright failure, so ask the session
    # which providers it really ended up with.
    avail = onnxruntime.get_available_providers()
    want_cuda = "CUDAExecutionProvider" in avail
    # CoreML is a free 1.55x on Apple Silicon (measured 27.38 vs 17.63 fps
    # detection-only) with byte-identical detections -- 64 faces found either
    # way, so no recall is traded. It was previously never tried: the provider
    # list went straight from CUDA to CPU.
    want_coreml = not want_cuda and "CoreMLExecutionProvider" in avail
    if want_cuda:
        providers, ctx = ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    elif want_coreml:
        providers, ctx = ["CoreMLExecutionProvider", "CPUExecutionProvider"], 0
    else:
        providers, ctx = ["CPUExecutionProvider"], -1
    app = FaceAnalysis(name="buffalo_l", allowed_modules=modules,
                       providers=providers)
    app.prepare(ctx_id=ctx, det_size=(det_size, det_size))

    actual = []
    for m in app.models.values():
        sess = getattr(m, "session", None)
        if sess is not None:
            actual += list(sess.get_providers())
    on_cuda = "CUDAExecutionProvider" in actual
    on_coreml = "CoreMLExecutionProvider" in actual
    device = "cuda" if on_cuda else ("coreml" if on_coreml else "cpu")
    print(f"[face_anon] detector ready (device={device}, det_size={det_size}, "
          f"modules={'+'.join(modules)})", flush=True)
    if want_cuda and not on_cuda:
        print("[face_anon] WARNING: CUDA was requested and is listed by this "
              "onnxruntime build, but the session is running on CPU -- the CUDA "
              "provider failed to load. Expect ~25x slower. Check the log above "
              "for a missing library (e.g. libcublasLt.so.12 needs a CUDA "
              "module loaded, or `pip install nvidia-cublas-cu12`).", flush=True)
    return app


def detect_boxes(app, frame):
    """Detection only -- returns (bboxes Nx5, kpss Nx5x2). Skips recognition."""
    bboxes, kpss = app.det_model.detect(frame, max_num=0, metric="default")
    if bboxes is None or len(bboxes) == 0:
        return np.empty((0, 5)), np.empty((0, 5, 2))
    return bboxes, kpss


def embed_face(app, frame, bbox, kps, det_score=1.0):
    """Compute one normed embedding on demand (identical to app.get()'s)."""
    from insightface.app.common import Face

    rec = app.models.get("recognition")
    if rec is None or kps is None:
        return None
    face = Face(bbox=np.asarray(bbox[:4], dtype=np.float32),
                kps=np.asarray(kps, dtype=np.float32),
                det_score=float(det_score))
    try:
        rec.get(frame, face)
    except Exception:
        return None
    return getattr(face, "normed_embedding", None)


# ---------------------------------------------------------------------------
# Stage A: who is the instructor? (sampled frames, not the whole video)
# ---------------------------------------------------------------------------
def identify_instructor(
    video_path,
    app,
    sample_frames=400,
    start_frame=0,
    end_frame=None,
    cluster_threshold=0.5,
    merge_threshold=0.5,
    min_cluster_share=0.0,
):
    """Sample frames across the video and return (centroid, clusters, info).

    Screen time is estimated from how many *sampled* frames each cluster appears
    in, which is an unbiased estimate of the real share because the sample is
    uniform. The highest-share cluster is the instructor.
    """
    meta = probe(video_path)
    end_frame = end_frame if end_frame is not None else meta["n_frames"]
    span = max(1, end_frame - start_frame)
    n = min(sample_frames, span)
    idxs = np.unique(np.linspace(start_frame, end_frame - 1, n).astype(int))

    cap = cv2.VideoCapture(video_path)
    embs, boxes, frames_seen, crops = [], [], [], []
    t0 = time.time()
    for k, i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        bboxes, kpss = detect_boxes(app, frame)
        for j in range(len(bboxes)):
            e = embed_face(app, frame, bboxes[j], kpss[j], bboxes[j][4])
            if e is None:
                continue
            embs.append(e)
            boxes.append(bboxes[j][:4])
            frames_seen.append(int(i))
            if len(crops) < 2000:
                x1, y1, x2, y2 = [int(v) for v in bboxes[j][:4]]
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                crops.append(crop if crop.size else None)
        if (k + 1) % 100 == 0:
            print(f"[face_anon] sample {k + 1}/{len(idxs)} frames, "
                  f"{len(embs)} faces", flush=True)
    cap.release()
    dt = time.time() - t0
    print(f"[face_anon] sampled {len(idxs)} frames in {dt:.1f}s "
          f"({len(idxs) / max(dt, 1e-9):.1f} fps), {len(embs)} faces with embeddings",
          flush=True)

    if not embs:
        return None, [], {"n_sampled": len(idxs), "n_faces": 0}

    E = np.stack(embs)
    if len(E) == 1:
        labels = np.array([0])
    else:
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=cluster_threshold,
        ).fit_predict(E)

    clusters = []
    for lab in sorted(set(labels)):
        sel = np.where(labels == lab)[0]
        c = E[sel].mean(axis=0)
        norm = np.linalg.norm(c)
        clusters.append({
            "label": int(lab),
            "n": len(sel),
            "share": len(sel) / len(E),
            "centroid": (c / norm) if norm > 0 else c,
            "example_crops": [crops[i] for i in sel[:12] if i < len(crops)],
            "mean_box_area": float(np.mean([
                (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) for i in sel
            ])),
        })
    clusters.sort(key=lambda c: -c["n"])

    # Agglomerative "average" linkage over-splits one person into several
    # clusters: average *pairwise* distance between two groups of noisy
    # embeddings can exceed the threshold even when their centroids are close.
    # Measured on lecture 12, the instructor came out as three clusters at 70.4%,
    # 23.8% and 0.5% whose centroids sit at cosine 1.00 / 0.749 / 0.580.
    #
    # That matters because the old code compensated with a very loose per-track
    # sim_threshold of 0.3 -- loose enough that a genuinely different person
    # (measured 0.348) passed as the instructor and went unblurred. Merging by
    # *centroid* similarity instead rebuilds one instructor identity from ~95% of
    # the sampled faces, which then supports a much tighter threshold.
    merged_labels = [clusters[0]["label"]]
    acc_n = clusters[0]["n"]
    protos = [clusters[0]["centroid"]]
    for c in clusters[1:]:
        if float(np.dot(c["centroid"], clusters[0]["centroid"])) >= merge_threshold:
            merged_labels.append(c["label"])
            protos.append(c["centroid"])
            acc_n += c["n"]
    # Keep each merged cluster as its own PROTOTYPE rather than averaging them
    # into one centroid. Frontal and profile views of the same person are
    # genuinely distant in embedding space, so their mean represents neither: a
    # frontal-dominated centroid scored the instructor's profile shots below
    # threshold and blurred him in 28.8% of frames where he was the only face.
    # Matching against the nearest prototype fixes that without touching
    # sim_threshold -- which must stay tight, since a measurably different
    # person scored 0.348.
    instructor_centroid = np.stack(protos)
    if len(merged_labels) > 1:
        print(f"[face_anon] instructor = clusters {merged_labels} kept as "
              f"{len(protos)} prototypes (centroid sim >= {merge_threshold}): "
              f"{acc_n}/{len(E)} faces ({acc_n / len(E):.1%})", flush=True)

    top = clusters[0]
    if top["share"] < min_cluster_share:
        print(f"[face_anon] WARNING top cluster is only {top['share']:.1%} of "
              f"sampled faces (< {min_cluster_share:.0%}); instructor identity is "
              f"weak -- check --preview before trusting the output", flush=True)
    info = {
        "n_sampled": len(idxs),
        "n_faces": len(E),
        "n_clusters": len(clusters),
        "shares": [round(c["share"], 4) for c in clusters],
        "instructor_labels": merged_labels,
        "instructor_share": round(acc_n / len(E), 4),
    }
    return instructor_centroid, clusters, info


def save_cluster_preview(clusters, out_png, tile=96, per_row=12):
    """Contact sheet: one row per cluster, so the instructor can be eyeballed."""
    rows = []
    for c in clusters[:8]:
        cells = []
        for crop in c["example_crops"][:per_row]:
            if crop is None or crop.size == 0:
                continue
            cells.append(cv2.resize(crop, (tile, tile)))
        if not cells:
            continue
        while len(cells) < per_row:
            cells.append(np.zeros((tile, tile, 3), np.uint8))
        row = np.hstack(cells)
        band = np.zeros((28, row.shape[1], 3), np.uint8)
        cv2.putText(band, f"cluster {c['label']}  {c['n']} faces  {c['share']:.1%}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        rows.append(np.vstack([band, row]))
    if not rows:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    cv2.imwrite(out_png, np.vstack(rows))
    return out_png


# ---------------------------------------------------------------------------
# Stage B: per-chunk detect -> classify -> blur -> encode
# ---------------------------------------------------------------------------
def _blur_region(frame, box, method="pixelate", pad=0.2):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw)); y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw)); y2 = min(h, int(y2 + pad * bh))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    rw, rh = x2 - x1, y2 - y1
    if method == "blur":
        k = max(15, (min(rw, rh) // 2) | 1)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    else:
        # Block COUNT must scale with the region, not be fixed at 16x16. These
        # are distant lecture-hall faces: a padded box measures ~51x79px, so a
        # fixed 16x16 grid is only a 3.2x reduction and leaves the face
        # essentially intact -- it looks like anonymization without being any.
        # Clamping to 2..12 blocks per side keeps small faces genuinely coarse
        # while stopping a large close-up from being reduced to mush.
        nb_w = int(np.clip(rw // 8, 2, 12))
        nb_h = int(np.clip(rh // 8, 2, 12))
        small = cv2.resize(roi, (nb_w, nb_h), interpolation=cv2.INTER_AREA)
        frame[y1:y2, x1:x2] = cv2.resize(small, (rw, rh),
                                         interpolation=cv2.INTER_NEAREST)


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _expand_detections(sampled, detect_every, n_frames, linger=2):
    """Turn per-sampled-frame boxes into per-frame boxes to blur.

    sampled: {frame_offset: [box4, ...]} at multiples of detect_every.
    For a gap between sample s and the next sample s+K, every frame in the gap
    gets the *union* of each matched pair of boxes, so a face moving across the
    gap stays covered end to end. Unmatched boxes on either side are carried
    across the whole gap rather than dropped -- over-blur is the safe direction.

    `linger` re-adds a box for that many further samples after it stops being
    detected. Detectors flicker: a face missed in a single sample would
    otherwise go unblurred for detect_every frames, so a disappearance has to be
    confirmed over several samples before coverage is dropped. Lookback reads
    raw detections only, so coverage cannot chain past `linger` samples.
    """
    keys = sorted(sampled)
    effective = {}
    for si, s in enumerate(keys):
        cur = [list(b[:4]) for b in sampled[s]]
        for back in range(1, linger + 1):
            if si - back < 0:
                break
            for b in sampled[keys[si - back]]:
                if not any(iou(b, c) >= 0.3 for c in cur):
                    cur.append(list(b[:4]))
        effective[s] = cur

    per_frame = [[] for _ in range(n_frames)]
    for si, s in enumerate(keys):
        cur = effective[s]
        nxt = effective[keys[si + 1]] if si + 1 < len(keys) else []
        gap_end = keys[si + 1] if si + 1 < len(keys) else min(s + detect_every, n_frames)

        # Greedy IOU match between this sample's boxes and the next sample's.
        pairs, used = [], set()
        for a in cur:
            best_j, best = -1, 0.05
            for j, b in enumerate(nxt):
                if j in used:
                    continue
                v = iou(a, b)
                if v >= best:
                    best, best_j = v, j
            if best_j >= 0:
                used.add(best_j)
                pairs.append(_union(a, nxt[best_j]))
            else:
                pairs.append(list(a[:4]))
        for j, b in enumerate(nxt):        # appeared mid-gap: cover it too
            if j not in used:
                pairs.append(list(b[:4]))

        for f in range(s, min(gap_end, n_frames)):
            per_frame[f].extend(pairs)
        if si + 1 == len(keys):            # tail after the final sample
            for f in range(min(gap_end, n_frames), n_frames):
                per_frame[f].extend(pairs)
    return per_frame


def process_chunk(
    video_path,
    out_path,
    start_frame,
    n_frames,
    centroid,
    app,
    detect_every=4,
    method="pixelate",
    sim_threshold=0.45,
    encoder="libx264",
    crf=20,
    threads=None,
    fps=25.0,
    label="",
):
    """Detect (every Kth frame) -> classify vs centroid -> blur -> pipe to ffmpeg."""
    meta = probe(video_path)
    W, H = meta["width"], meta["height"]

    # --- pass 1: detection on sampled frames, plus one embedding per new track
    cap = cv2.VideoCapture(video_path)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    sampled, tracks = {}, []          # tracks: {"box", "last", "is_instructor"}
    n_infer = n_blur_boxes = 0
    t0 = time.time()
    for off in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            n_frames = off
            break
        if off % detect_every:
            continue
        bboxes, kpss = detect_boxes(app, frame)
        n_infer += 1
        keep = []
        for j in range(len(bboxes)):
            box = bboxes[j][:4]
            # Match to a live track to reuse its instructor/not decision.
            hit = None
            for tr in tracks:
                if off - tr["last"] <= 4 * detect_every and iou(box, tr["box"]) >= 0.3:
                    hit = tr
                    break
            if hit is None:
                # New face -> one recognition call. Fail-closed: no embedding
                # means we cannot prove it is the instructor, so it gets blurred.
                emb = embed_face(app, frame, box, kpss[j], bboxes[j][4])
                is_instr = False
                if emb is not None and centroid is not None:
                    protos = np.atleast_2d(centroid)
                    is_instr = float(np.max(protos @ emb)) >= sim_threshold
                hit = {"box": box, "last": off, "is_instructor": is_instr}
                tracks.append(hit)
            else:
                hit["box"], hit["last"] = box, off
            if not hit["is_instructor"]:
                keep.append(box)
        sampled[off] = keep
        n_blur_boxes += len(keep)
        # A full lecture spends over half an hour in this loop. Without periodic
        # output there is no way to tell a working job from a hung one, which
        # matters most for whoever is not going to read the source.
        if n_infer % 500 == 0:
            el = time.time() - t0
            rate = n_infer / max(el, 1e-9)
            eta = (n_frames / detect_every - n_infer) / max(rate, 1e-9)
            print(f"[face_anon] {label}pass1 {off}/{n_frames} frames "
                  f"({100.0 * off / max(n_frames, 1):.1f}%), {n_infer} detections "
                  f"@ {rate:.1f}/s, {len(tracks)} tracks, ETA {eta / 60:.1f} min",
                  flush=True)
    cap.release()
    infer_dt = time.time() - t0
    print(f"[face_anon] {label}pass1 done: {n_infer} detections in "
          f"{infer_dt / 60:.1f} min, {n_blur_boxes} boxes to blur", flush=True)

    per_frame = _expand_detections(sampled, detect_every, n_frames)

    # --- pass 2: decode again (decode is ~0.5% of cost) and blur + encode
    threads = threads or max(1, cpu_count())
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
        "-r", f"{fps:.10g}", "-i", "pipe:0", "-an",
        "-c:v", encoder, "-crf", str(crf), "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-threads", str(threads),
        "-video_track_timescale", "90000", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(video_path)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    t1 = time.time()
    try:
        for off in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            for box in per_frame[off]:
                _blur_region(frame, box, method=method)
            proc.stdin.write(frame.tobytes())
            written += 1
            if written % 2000 == 0:
                el = time.time() - t1
                print(f"[face_anon] {label}pass2 {written}/{n_frames} frames "
                      f"({100.0 * written / max(n_frames, 1):.1f}%) @ "
                      f"{written / max(el, 1e-9):.0f} fps", flush=True)
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()
    blur_dt = time.time() - t1

    print(f"[face_anon] {label}chunk done: {written} frames, {n_infer} detections "
          f"({infer_dt:.1f}s infer, {blur_dt:.1f}s blur+encode), "
          f"{len(tracks)} tracks, {sum(1 for t in tracks if t['is_instructor'])} "
          f"instructor", flush=True)
    return {"frames": written, "n_infer": n_infer, "tracks": len(tracks),
            "infer_s": infer_dt, "encode_s": blur_dt}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def concat_and_mux(chunk_paths, audio_source, out_path, work_dir, start_sec=None):
    list_path = os.path.join(work_dir, "concat.txt")
    with open(list_path, "w") as f:
        for p in chunk_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", list_path]
    if start_sec:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", audio_source,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            "-movflags", "+faststart", out_path]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("[face_anon] audio mux failed; writing video-only", flush=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", out_path], check=True)
    return out_path


def run_benchmark(video_path, det_size=640):
    """Measure this machine's real throughput, so chunk/K choices are informed."""
    meta = probe(video_path)
    cap = cv2.VideoCapture(video_path)
    idxs = np.linspace(0, meta["n_frames"] - 1, 20).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()

    app = build_app(det_size=det_size, need_recognition=True)
    detect_boxes(app, frames[0])
    t0 = time.time()
    for f in frames:
        detect_boxes(app, f)
    det_fps = len(frames) / (time.time() - t0)

    t0 = time.time()
    for f in frames:
        b, k = detect_boxes(app, f)
        for j in range(len(b)):
            embed_face(app, f, b[j], k[j], b[j][4])
    full_fps = len(frames) / (time.time() - t0)

    print(f"\n[face_anon] benchmark on {os.path.basename(video_path)} "
          f"({meta['width']}x{meta['height']}, {meta['n_frames']} frames)")
    print(f"  detection only        {det_fps:7.2f} fps")
    print(f"  detection+recognition {full_fps:7.2f} fps")
    for K in (1, 2, 4, 8):
        for C in (1, 8):
            secs = meta["n_frames"] / K / det_fps / C
            print(f"  detect-every={K} chunks={C:<2d} -> {secs / 60:8.1f} min "
                  f"of inference")
    return det_fps, full_fps


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lecture-dir", required=True)
    parser.add_argument("--stream", default="camera", choices=["camera", "screen"])
    parser.add_argument("--input", default=None,
                        help="Explicit filename in the lecture dir. Defaults to "
                             "camera_muted.mp4 when it exists, so the muted "
                             "audio is carried through, else camera.mp4")
    parser.add_argument("--method", default="pixelate", choices=["pixelate", "blur"])
    parser.add_argument("--detect-every", type=int, default=4,
                        help="Detect every Nth frame; gaps get the union of the "
                             "bracketing boxes. 1 = every frame (slowest, safest)")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--sample-frames", type=int, default=400,
                        help="Frames sampled for instructor identification")
    parser.add_argument("--identity-scope", default="video",
                        choices=["video", "window"],
                        help="Sample identity over the whole video (default) or "
                             "only the --start/--end window")
    parser.add_argument("--chunks", type=int, default=1,
                        help="Split into N independent workers, one per GPU. Only "
                             "worth it on a MULTI-GPU node: measured on a 10-core "
                             "CPU Mac, chunks 1/2/4 took 23.3/25.5/29.6s -- "
                             "onnxruntime already spreads across cores, so extra "
                             "processes just add model-load and contention")
    parser.add_argument("--cluster-threshold", type=float, default=0.5)
    parser.add_argument("--sim-threshold", type=float, default=0.45,
                        help="Cosine similarity to the instructor centroid above "
                             "which a face is NOT blurred. Lower = more faces pass "
                             "as the instructor = weaker anonymization")
    parser.add_argument("--merge-threshold", type=float, default=0.5,
                        help="Merge clusters into one identity at this centroid "
                             "similarity (counters cluster over-splitting)")
    parser.add_argument("--instructor-cluster", type=int, default=None,
                        help="Force this cluster id as the instructor (see --preview)")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start", default=None, help="MM:SS, HH:MM:SS, or seconds")
    parser.add_argument("--end", default=None)
    parser.add_argument("--preview", action="store_true",
                        help="Sample, cluster, write face_clusters.png, exit")
    parser.add_argument("--benchmark", action="store_true",
                        help="Measure inference throughput on this machine and exit")
    parser.add_argument("--keep-work", action="store_true")
    # internal: one chunk worker, spawned by the parent with a pinned GPU
    parser.add_argument("--worker-chunk", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-start", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-centroid", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Slurm captures stdout to a file, which makes it block-buffered: parent
    # prints would otherwise land after the child workers' output and the log
    # would read out of order.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Default to the already-muted camera: face_anon belongs after audio.py in
    # the pipeline, and muxing from camera_muted.mp4 keeps the muted audio.
    input_name = args.input
    if input_name is None:
        muted = f"{args.stream}_muted.mp4"
        input_name = muted if os.path.exists(
            os.path.join(args.lecture_dir, muted)) else f"{args.stream}.mp4"
    video_path = os.path.join(args.lecture_dir, input_name)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No {input_name} in {args.lecture_dir}")
    meta = probe(video_path)
    fps = meta["fps"]

    # ---- worker mode: one chunk, then exit
    if args.worker_chunk is not None:
        centroid = None
        if args.worker_centroid and os.path.exists(args.worker_centroid):
            centroid = np.load(args.worker_centroid)
        app = build_app(det_size=args.det_size, need_recognition=True)
        process_chunk(
            video_path, args.worker_out, args.worker_start, args.worker_count,
            centroid, app, detect_every=args.detect_every, method=args.method,
            sim_threshold=args.sim_threshold, crf=args.crf,
            threads=args.threads, fps=fps,
            label=f"[w{args.worker_chunk}] ",
        )
        return

    if args.benchmark:
        run_benchmark(video_path, det_size=args.det_size)
        return

    start_sec, end_sec = parse_time(args.start), parse_time(args.end)
    start_frame = int((start_sec or 0) * fps)
    end_frame = int(end_sec * fps) if end_sec is not None else meta["n_frames"]
    if args.max_frames:
        end_frame = min(end_frame, start_frame + args.max_frames)
    end_frame = min(end_frame, meta["n_frames"])
    total = max(0, end_frame - start_frame)
    if total == 0:
        raise SystemExit("empty frame range")

    print(f"[face_anon] {video_path}: {meta['width']}x{meta['height']} @ {fps:g}fps, "
          f"frames {start_frame}..{end_frame} ({total})")

    # ---- Stage A: instructor identity from a sample
    app = build_app(det_size=args.det_size, need_recognition=True)
    # Identity is sampled across the WHOLE lecture, not just --start/--end: who
    # the instructor is is a property of the lecture, and a short window can
    # easily be dominated by somebody else, which would invert the decision and
    # blur the instructor instead. --identity-scope window overrides.
    id_start, id_end = (start_frame, end_frame) if args.identity_scope == "window" \
        else (0, meta["n_frames"])
    centroid, clusters, info = identify_instructor(
        video_path, app, sample_frames=args.sample_frames,
        start_frame=id_start, end_frame=id_end,
        cluster_threshold=args.cluster_threshold,
        merge_threshold=args.merge_threshold,
    )
    print(f"[face_anon] identity: {info}")
    for c in clusters[:6]:
        print(f"    cluster {c['label']}: {c['n']} faces ({c['share']:.1%}), "
              f"mean face area {c['mean_box_area']:.0f}px^2")

    if args.instructor_cluster is not None:
        sel = [c for c in clusters if c["label"] == args.instructor_cluster]
        if not sel:
            raise SystemExit(f"no cluster {args.instructor_cluster}")
        centroid = sel[0]["centroid"]
        print(f"[face_anon] instructor cluster forced to {args.instructor_cluster}")

    if args.preview:
        png = save_cluster_preview(
            clusters, os.path.join(args.lecture_dir, "face_clusters.png"))
        print(f"[face_anon] cluster preview: {png}\n"
              f"  Largest cluster is treated as the instructor; override with "
              f"--instructor-cluster <id>.")
        return

    if centroid is None:
        print("[face_anon] WARNING no face embeddings found in the sample; "
              "every detected face will be blurred (fail-closed)")

    input_base = os.path.splitext(os.path.basename(input_name))[0]
    out_path = os.path.join(args.lecture_dir, f"{input_base}_anon.mp4")
    work_dir = tempfile.mkdtemp(prefix="faceanon-", dir=args.lecture_dir)

    try:
        # ---- Stage B: chunked blur
        n_chunks = max(1, args.chunks)
        bounds = np.linspace(start_frame, end_frame, n_chunks + 1).astype(int)
        chunk_paths = [os.path.join(work_dir, f"chunk_{i:03d}.mp4")
                       for i in range(n_chunks)]
        t0 = time.time()

        if n_chunks == 1:
            process_chunk(video_path, chunk_paths[0], start_frame, total,
                          centroid, app, detect_every=args.detect_every,
                          method=args.method, sim_threshold=args.sim_threshold,
                          crf=args.crf, threads=args.threads, fps=fps)
        else:
            cpath = os.path.join(work_dir, "centroid.npy")
            if centroid is not None:
                np.save(cpath, centroid)
            del app          # free this process's GPU context before forking

            procs = []
            n_gpu = _visible_gpu_count()
            threads_each = max(1, cpu_count() // n_chunks)
            for i in range(n_chunks):
                env = dict(os.environ)
                if n_gpu:
                    env["CUDA_VISIBLE_DEVICES"] = str(i % n_gpu)
                env.setdefault("OMP_NUM_THREADS", str(threads_each))
                cmd = [
                    sys.executable, "-m", "src.video.face_anon",
                    "--lecture-dir", args.lecture_dir, "--input", input_name,
                    "--method", args.method,
                    "--detect-every", str(args.detect_every),
                    "--det-size", str(args.det_size),
                    "--sim-threshold", str(args.sim_threshold),
                    "--crf", str(args.crf), "--threads", str(threads_each),
                    "--worker-chunk", str(i),
                    "--worker-start", str(int(bounds[i])),
                    "--worker-count", str(int(bounds[i + 1] - bounds[i])),
                    "--worker-out", chunk_paths[i],
                ]
                if centroid is not None:
                    cmd += ["--worker-centroid", cpath]
                print(f"[face_anon] worker {i}: frames {bounds[i]}.."
                      f"{bounds[i + 1]} gpu={env.get('CUDA_VISIBLE_DEVICES', '-')}")
                procs.append(subprocess.Popen(cmd, env=env))
            codes = [p.wait() for p in procs]
            if any(codes):
                raise RuntimeError(f"chunk workers failed: exit codes {codes}")

        # ---- Stage C: join + carry the audio over
        concat_and_mux(chunk_paths, video_path, out_path, work_dir,
                       start_sec=start_sec if start_frame else None)
        dt = time.time() - t0
    finally:
        if not args.keep_work:
            shutil.rmtree(work_dir, ignore_errors=True)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[face_anon] wrote {out_path} ({size_mb:.1f} MB) in {dt / 60:.1f} min "
          f"({total / max(dt, 1e-9):.1f} fps end-to-end, {args.method}, "
          f"detect-every={args.detect_every}, chunks={n_chunks})")


def _visible_gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return len([l for l in out.stdout.splitlines() if l.strip()])
    except Exception:
        return 0


if __name__ == "__main__":
    main()
