"""Vision tier: is the instructor in shot, how big, and how many others.

Uses face_anon's own detector via `build_app`, deliberately. The scanner is
predicting what the anonymization stage will have to do, so it should be
looking through the same lens -- same buffalo_l models, same provider
selection, same CoreML path on Apple Silicon.

What this is NOT is a decision about who the instructor is. face_anon's
`identify_instructor` does that properly, with dense opening sampling and
centroid-merged prototypes, because getting it backwards means blurring the
lecturer and publishing the students. The clustering here is a cheap estimate
over a couple of hundred frames, and it exists to answer three planning
questions:

    is there a person in shot at all, and for how much of the lecture
    how big are they -- the brand rail is 432px against a 1920px source
    how many other faces will have to be pixelated, and how often

Get any of those badly wrong and the cost is a misranked lecture, not a privacy
failure. Nothing downstream consumes this; `face_anon --preview` remains the
thing to run before committing to an anonymization pass.
"""

import numpy as np

from src.scan.media import iter_frames, probe

# Frames to run detection on. 200 across 90 minutes is one every 27 seconds --
# enough for a presence fraction to converge, and about 12s of inference on an
# M5 at the measured 17.5 fps detection-only.
SAMPLE_FRAMES = 200

# Detection runs on frames this size. Small enough to decode a whole lecture
# quickly, large enough that a lecturer at the far end of a hall is still
# several dozen pixels of face.
DETECT_W, DETECT_H = 1280, 720

# Same clustering constants face_anon uses, so the estimate here and the real
# pass later at least agree about what "one person" means.
CLUSTER_THRESHOLD = 0.5
MERGE_THRESHOLD = 0.5

# A cluster seen in fewer than this many sampled frames is a detection artefact
# -- a face in a photo on a slide, a reflection -- not a person in the room.
MIN_CLUSTER_FRAMES = 3


def _cluster(embeddings):
    """Agglomerative clustering on cosine distance, as face_anon does it."""
    from sklearn.cluster import AgglomerativeClustering
    if len(embeddings) == 1:
        return np.zeros(1, dtype=int)
    return AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=CLUSTER_THRESHOLD).fit_predict(np.asarray(embeddings))


def measure(camera_path, duration_s, app=None, sample_frames=SAMPLE_FRAMES,
            verbose=False):
    """Instructor presence and the anonymization load, from sampled frames."""
    m = {}
    info = probe(camera_path)
    if not info or not info.get("has_video"):
        return m

    if app is None:
        from src.video.face_anon import build_app
        app = build_app(det_size=640, need_recognition=True, quiet=True)

    # Decode the keyframe grid, then keep every Nth so the detector sees an
    # evenly spread sample rather than the first few minutes.
    total_est = max(int(duration_s / 2.4), 1)
    stride = max(1, total_est // sample_frames)

    faces_per_frame = []
    embeddings, box_heights, frame_of = [], [], []
    n_frames = 0
    for idx, rgb in iter_frames(camera_path, DETECT_W, DETECT_H,
                                pix_fmt="rgb24", stride=stride):
        bgr = rgb[:, :, ::-1]           # insightface expects BGR, as cv2 gives
        try:
            faces = app.get(bgr)
        except Exception:               # a bad frame must not kill the scan
            continue
        faces_per_frame.append(len(faces))
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue
            x1, y1, x2, y2 = f.bbox
            embeddings.append(emb)
            box_heights.append(float(y2 - y1))
            frame_of.append(n_frames)
        n_frames += 1

    if n_frames == 0:
        return m
    fpf = np.asarray(faces_per_frame)
    m["vision_frames_sampled"] = int(n_frames)
    m["face_detected_pct"] = float(np.mean(fpf > 0) * 100.0)
    m["multi_face_pct"] = float(np.mean(fpf > 1) * 100.0)
    m["mean_faces_per_frame"] = float(fpf.mean())

    if len(embeddings) < 2:
        # A lecture where almost no face was found is a real finding, not a
        # missing measurement: report presence as measured and let the
        # instructor_visible gate act on it.
        m["instructor_in_frame_pct"] = m["face_detected_pct"]
        m["student_face_clusters"] = 0
        m["student_face_pct"] = 0.0
        return m

    labels = _cluster(embeddings)
    frame_of = np.asarray(frame_of)
    box_heights = np.asarray(box_heights)

    clusters = []
    for lab in sorted(set(labels)):
        sel = np.flatnonzero(labels == lab)
        seen = np.unique(frame_of[sel])
        centroid = np.asarray(embeddings)[sel].mean(axis=0)
        norm = np.linalg.norm(centroid)
        clusters.append({
            "label": int(lab),
            "faces": int(sel.size),
            "frames": int(seen.size),
            "centroid": centroid / norm if norm > 0 else centroid,
            "median_box_h": float(np.median(box_heights[sel])),
        })
    clusters = [c for c in clusters if c["frames"] >= MIN_CLUSTER_FRAMES]
    if not clusters:
        m["instructor_in_frame_pct"] = m["face_detected_pct"]
        m["student_face_clusters"] = 0
        m["student_face_pct"] = 0.0
        return m
    clusters.sort(key=lambda c: -c["frames"])

    # The instructor is the person on screen most, plus any cluster whose
    # centroid is close enough to be the same person seen from another angle.
    # Agglomerative average linkage over-splits one face into several; face_anon
    # documents the same thing (one person came out as three clusters at 70.4%,
    # 23.8% and 0.5%), and not merging them here would report a lecturer's
    # profile view as a second student.
    lead = clusters[0]
    inst = [c for c in clusters
            if float(np.dot(c["centroid"], lead["centroid"])) >= MERGE_THRESHOLD]
    inst_labels = {c["label"] for c in inst}

    inst_mask = np.isin(labels, list(inst_labels))
    inst_frames = np.unique(frame_of[inst_mask])
    m["instructor_in_frame_pct"] = float(inst_frames.size / n_frames * 100.0)

    # Frames with ANY non-instructor face. The headline student-exposure
    # number: multi_face_pct needs two faces at once and so misses the camera
    # that cuts to a single student asking a question, which exposes that
    # student just as completely.
    student_frames = np.unique(frame_of[~inst_mask])
    m["student_face_pct"] = float(student_frames.size / n_frames * 100.0)

    inst_heights = box_heights[np.isin(labels, list(inst_labels))]
    # Report at a 1080p equivalent so the number means the same thing whether
    # the source was 720p or 1080p -- the brand rail scales from 1920 wide.
    scale = 1080.0 / DETECT_H
    m["instructor_face_px"] = float(np.median(inst_heights) * scale)

    others = [c for c in clusters if c["label"] not in inst_labels]
    m["student_face_clusters"] = len(others)
    m["face_clusters_total"] = len(clusters)
    m["instructor_cluster_parts"] = len(inst)
    if verbose:
        print(f"[scan] {n_frames} frames, {len(embeddings)} faces, "
              f"{len(clusters)} clusters, instructor in "
              f"{m['instructor_in_frame_pct']:.0f}% at "
              f"{m['instructor_face_px']:.0f}px")
    return m
