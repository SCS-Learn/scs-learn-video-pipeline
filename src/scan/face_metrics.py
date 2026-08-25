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

Student exposure is the highest-weighted thing measured here, so the accounting
for it is spelled out rather than left to fall out of the arithmetic. Every
detection lands in exactly one of three buckets -- instructor, student, or
artefact -- and `student_face_pct` and `student_face_clusters` are both derived
from the student bucket, so they cannot disagree. See `_classify_clusters` for
why the merge has to happen before the artefact filter and not after.
"""

import numpy as np

from src.scan.media import iter_frames, keyframe_times, probe

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

# A cluster seen in fewer than this many sampled frames is not by itself
# evidence of a person in the room. It is NOT a licence to discard it: see
# `_looks_like_artefact`, which wants positive evidence before dropping one.
MIN_CLUSTER_FRAMES = 3

# Below this detection score the detector itself was unsure. FaceAnalysis
# accepts at det_thresh=0.5 and real faces in these lectures sit at 0.75-0.95,
# so this takes only the bottom sliver of what was accepted -- the band where
# posters, printed faces on a slide and reflections live.
ARTEFACT_DET_SCORE = 0.65

# A box that moves less than this many pixels (at DETECT_W x DETECT_H) between
# samples taken tens of seconds apart is not attached to a body.
STATIC_TOL_PX = 3.0

# Panopto's GOP. Only a fallback now -- the real keyframe count is measured.
FALLBACK_GOP_S = 2.4


def _cluster(embeddings):
    """Agglomerative clustering on cosine distance, as face_anon does it."""
    from sklearn.cluster import AgglomerativeClustering
    if len(embeddings) == 1:
        return np.zeros(1, dtype=int)
    return AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=CLUSTER_THRESHOLD,
    ).fit_predict(np.asarray(embeddings))


def _sample_stride(camera_path, duration_s, sample_frames):
    """Frames to skip so `sample_frames` samples span the whole lecture.

    `iter_frames` walks the keyframe grid, so the population being sampled is
    the KEYFRAMES, not the frames. This used to be estimated as
    duration / 2.4 -- Panopto's GOP, hardcoded -- which silently mis-sizes the
    stride on anything else: a 6s GOP has 2.5x fewer keyframes than assumed, so
    the stride comes out 2.5x too large and a request for 200 frames delivers
    about 89. Under-delivering is the worse direction, because every fraction
    here is measured against however many frames actually arrived.

    Counting them costs one ffprobe (`media.keyframe_times`) and no second
    decode pass. On 15-210 lecture 12 the measured count is 1991 against the
    1989 the 2.4s assumption gave, i.e. the same stride of 9 and the same 222
    sampled frames -- the assumption was right on this corpus and wrong in
    general, which is exactly the kind of thing that survives a whole project.
    """
    total = len(keyframe_times(camera_path))
    note = None
    if total < 2:
        # ffprobe failed, or the file genuinely has no usable grid. Fall back
        # to the old assumption rather than refusing to measure the lecture.
        total = max(int(duration_s / FALLBACK_GOP_S), 1)
        note = ("keyframe grid unreadable; assuming a "
                f"{FALLBACK_GOP_S}s GOP for the sampling stride")
    return max(1, total // max(sample_frames, 1)), total, note


def _looks_like_artefact(c):
    """Positive evidence that a short-lived cluster is not a person at all.

    Only ever consulted for clusters below MIN_CLUSTER_FRAMES, and it has to
    find a REASON -- absence of evidence leaves the cluster counted as a
    student. That asymmetry is the point: a genuine student who appears in one
    or two sampled frames is real exposure (at one sample per ~27 seconds, a
    camera that cuts to a questioner for half a minute produces exactly one
    such frame), while the things MIN_CLUSTER_FRAMES was written for -- a face
    printed on a slide, a poster on the back wall, a reflection in the
    projection screen -- are not.

    Two signals, either sufficient:

    * The detector was unsure. A photographed, projected or reflected face
      scores near the accept threshold where a person in the room does not.
    * The box never moved. Scenery is pinned to a pixel; a person sampled tens
      of seconds apart is not. Needs two frames to say anything, so it can
      never rescue us from a one-frame poster -- the score test covers that.

    Residual risk, stated plainly: a real student who is far away, motion
    blurred, or half-turned can score under ARTEFACT_DET_SCORE and be dropped,
    and this metric would then under-report a lecture that does expose someone.
    Three things bound it. The drop only applies below MIN_CLUSTER_FRAMES, so
    it can move `student_face_pct` by at most a couple of sampled frames --
    under 1.5% on a 200-frame sample, against a metric whose band runs to 40%.
    `face_detected_pct`, `multi_face_pct` and `mean_faces_per_frame` count raw
    detections with no clustering at all, so the evidence stays on the record
    even when this call goes the wrong way. And every drop is counted into
    `artefact_face_clusters` rather than vanishing, so a lecture that leans on
    this judgement is visible as one.
    """
    if c["frames"] >= MIN_CLUSTER_FRAMES:
        return False
    if c["det_score"] is not None and c["det_score"] < ARTEFACT_DET_SCORE:
        return True
    if c["frames"] >= 2 and c["motion_px"] is not None \
            and c["motion_px"] <= STATIC_TOL_PX:
        return True
    return False


def _classify_clusters(clusters):
    """Split clusters into (instructor, student, artefact).

    The ORDER of the two operations here is the whole subtlety, and getting it
    backwards was a real bug: the artefact filter used to run first, over the
    cluster list, while the instructor merge and the student count ran against
    the unfiltered labels. Anything the filter dropped was therefore neither
    instructor nor rejected -- it fell through into "not the instructor" and
    was reported as student exposure by a module that had just printed zero
    student clusters. On 15-210 lecture 12's real numbers that kind of phantom
    is worth about 1.13 points of final score, against a metric weighted 2.5
    inside a dimension worth 20% of the total.

    So: merge first, filter second.

    The merge runs over EVERY cluster, small ones included, because a small
    cluster is most often the lead person seen from another angle. Average
    linkage agglomeration over-splits one face -- face_anon reports the
    instructor
    coming out as three clusters at 70.4%, 23.8% and 0.5% -- and the 0.5% tail
    is exactly what a frames-based filter deletes. Reconsidering those for the
    instructor before rejecting anything is strictly better than dropping them:
    dropping loses their frames from `instructor_in_frame_pct` and, done in the
    wrong order, invents a student out of the lecturer's own profile view.
    `identify_instructor` merges over every cluster too, with no minimum at
    all, so this also stops the estimate and the real pass disagreeing.

    The lead is picked from clusters that clear MIN_CLUSTER_FRAMES, though.
    Whoever the instructor is, they are on screen for more than two sampled
    frames, and letting a one-frame detection become the lead would make a face
    on a slide the instructor and every real person in the room a student.
    """
    real = [c for c in clusters if c["frames"] >= MIN_CLUSTER_FRAMES]
    inst, others = [], []
    if real:
        lead = max(real, key=lambda c: (c["frames"], c["faces"]))
        for c in clusters:
            sim = float(np.dot(c["centroid"], lead["centroid"]))
            (inst if sim >= MERGE_THRESHOLD else others).append(c)
    else:
        # Nobody is on screen long enough to be the lecturer, so there is no
        # instructor for anyone to be "other" than. Every face still goes
        # through the artefact test below, and whatever passes it is counted as
        # a student rather than zeroed: a camera roving over a full room
        # produces exactly this shape -- many short-lived clusters and no
        # dominant one -- and calling that zero exposure would be the most
        # dangerous answer available.
        others = list(clusters)

    students = [c for c in others if not _looks_like_artefact(c)]
    artefacts = [c for c in others if _looks_like_artefact(c)]
    return inst, students, artefacts


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
    stride, n_keyframes, note = _sample_stride(
        camera_path, duration_s, sample_frames)
    if note and verbose:
        print(f"[scan] {note}")

    faces_per_frame = []
    embeddings, box_heights, box_cx, box_cy = [], [], [], []
    det_scores, frame_of = [], []
    n_frames = 0
    for _t, rgb in iter_frames(camera_path, DETECT_W, DETECT_H,
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
            box_cx.append(float(x1 + x2) / 2.0)
            box_cy.append(float(y1 + y2) / 2.0)
            score = getattr(f, "det_score", None)
            # None, not a default: a detector that reports no confidence must
            # not be read as reporting LOW confidence, or every detection it
            # made would look like an artefact and every student would vanish.
            det_scores.append(None if score is None else float(score))
            frame_of.append(n_frames)
        n_frames += 1

    if n_frames == 0:
        return m
    fpf = np.asarray(faces_per_frame)
    m["vision_frames_sampled"] = int(n_frames)
    m["vision_keyframes_total"] = int(n_keyframes)
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
        m["artefact_face_clusters"] = 0
        return m

    E = np.asarray(embeddings)
    labels = _cluster(E)
    frame_of = np.asarray(frame_of)
    box_heights = np.asarray(box_heights)
    box_cx = np.asarray(box_cx)
    box_cy = np.asarray(box_cy)

    clusters = []
    for lab in sorted(set(labels)):
        sel = np.flatnonzero(labels == lab)
        seen = np.unique(frame_of[sel])
        centroid = E[sel].mean(axis=0)
        norm = np.linalg.norm(centroid)
        scores = [det_scores[i] for i in sel]
        # How far the box wandered over the cluster's life, as the largest of
        # the three spreads. One number, and any real movement trips it.
        motion = float(max(np.ptp(box_cx[sel]), np.ptp(box_cy[sel]),
                           np.ptp(box_heights[sel]))) if sel.size > 1 else None
        clusters.append({
            "label": int(lab),
            "faces": int(sel.size),
            "frames": int(seen.size),
            "centroid": centroid / norm if norm > 0 else centroid,
            "median_box_h": float(np.median(box_heights[sel])),
            "det_score": (None if any(s is None for s in scores)
                          else float(np.mean(scores))),
            "motion_px": motion,
        })

    inst, students, artefacts = _classify_clusters(clusters)

    def _frames_of(group):
        """Distinct sampled frames in which this set of clusters appears."""
        if not group:
            return np.array([], dtype=int)
        mask = np.isin(labels, [c["label"] for c in group])
        return np.unique(frame_of[mask])

    if inst:
        m["instructor_in_frame_pct"] = float(
            _frames_of(inst).size / n_frames * 100.0)
    else:
        # No cluster was persistent enough to name an instructor. Presence is
        # still a real measurement, so report it as measured and let the
        # instructor_visible gate act on it -- knowingly counting the same
        # faces below as student exposure too, because when identity cannot be
        # established both readings have to stay on the table.
        m["instructor_in_frame_pct"] = m["face_detected_pct"]

    # Frames with a face belonging to a non-instructor PERSON. The headline
    # student-exposure number: multi_face_pct needs two faces at once and so
    # misses the camera that cuts to a single student asking a question, which
    # exposes that student just as completely. Counted from the same `students`
    # list as student_face_clusters below, so the two cannot drift apart --
    # they were derived independently once, and the module reported 2% exposure
    # next to 0 students.
    m["student_face_pct"] = float(_frames_of(students).size / n_frames * 100.0)

    if inst:
        inst_heights = box_heights[
            np.isin(labels, [c["label"] for c in inst])]
        # Report at a 1080p equivalent so the number means the same thing
        # whether the source was 720p or 1080p -- the brand rail scales from
        # 1920 wide. Left unset when no instructor was identified: an absent
        # metric reads as unmeasured downstream, a zero would read as a
        # lecturer with a 0px face.
        scale = 1080.0 / DETECT_H
        m["instructor_face_px"] = float(np.median(inst_heights) * scale)

    m["student_face_clusters"] = len(students)
    m["artefact_face_clusters"] = len(artefacts)
    m["face_clusters_total"] = len(inst) + len(students)
    m["instructor_cluster_parts"] = len(inst)

    # The invariant, checked rather than trusted. A percentage with no cluster
    # behind it is the bug this module shipped with, and it is undetectable
    # downstream -- the number is plausible, it just is not real. Raising here
    # surfaces as a scan error against that one lecture, which is loud; a wrong
    # exposure figure is silent and misranks the whole semester.
    if bool(students) != bool(m["student_face_pct"] > 0.0):
        raise ValueError(
            f"student exposure is self-contradictory: "
            f"{len(students)} student clusters but "
            f"{m['student_face_pct']:.2f}% of frames")

    if verbose:
        px = m.get("instructor_face_px")
        print(f"[scan] {n_frames} frames, {len(embeddings)} faces, "
              f"{len(clusters)} clusters ({len(inst)} instructor, "
              f"{len(students)} student, {len(artefacts)} artefact), "
              f"instructor in {m['instructor_in_frame_pct']:.0f}% at "
              f"{'?' if px is None else format(px, '.0f')}px")
        for c in artefacts:
            print(f"[scan]   dropped cluster {c['label']} as an artefact: "
                  f"{c['frames']} frame(s), det_score={c['det_score']}, "
                  f"motion={c['motion_px']}px")
    return m
