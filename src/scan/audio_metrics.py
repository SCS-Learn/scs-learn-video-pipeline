"""Audio measurement: is it audible, is it clean, and does the speaker vary.

Everything here comes off ONE decode (see media.decode_audio), which hands back
the ebur128 summary and the raw 16 kHz mono signal together. The standardised
loudness figures are ffmpeg's; the rest is computed here because the numbers
that matter for a lecture -- room tone, speech-to-noise, how monotone the
delivery is -- are not things astats reports in a usable form.

On astats specifically: its "Noise floor dB" reads -86 dBFS on 15-210 lecture
12, which is the quietest sample in the file, not the room tone anyone hears.
The floor that matters is the level the signal sits at between words, and that
is a percentile of short-frame energy, computed below.

The prosody pair is the interesting part. `pitch_variety_st` is the closest
thing to an objective handle on "is this lecturer engaging": the standard
deviation of voiced-frame F0, in semitones so that a bass and a tenor are
comparable. Monotone delivery really does measure under ~1.5 st and animated
delivery really does measure over ~3. It is not a personality test -- it is the
one thing about delivery a machine can measure honestly, and the rubric weights
it accordingly rather than pretending to judge charisma.
"""

import numpy as np

SR = 16000

# Short-term analysis grid for level work: 25 ms window, 10 ms hop. Standard
# speech-processing framing, fine enough to catch a dropped syllable.
LEVEL_WIN = int(0.025 * SR)
LEVEL_HOP = int(0.010 * SR)

# F0 analysis: a longer window, because the lowest pitch we look for (70 Hz)
# needs two periods inside it to autocorrelate at all.
F0_WIN = int(0.040 * SR)
F0_HOP = int(0.020 * SR)
F0_MIN_HZ, F0_MAX_HZ = 70.0, 350.0
F0_VOICED_R = 0.35          # autocorrelation peak height that counts as voiced
F0_MAX_FRAMES = 30000       # cap the work; 10 minutes of voiced speech is plenty

# Anything under this is the digital silence of an unstarted recording, not
# room tone, and including it drags every percentile down.
DIGITAL_SILENCE_DB = -75.0
# How far above the floor a frame must sit to count as speech.
VAD_MARGIN_DB = 8.0

# A word whose audio sits this close to the room-tone floor was not actually
# captured. Used by speech_metrics, which has the word timings.
AT_FLOOR_MARGIN_DB = 3.0


def _frame_rms_db(x, win, hop):
    """Per-frame RMS in dBFS, via a cumulative sum of squares."""
    if x.size < win:
        return np.zeros(0, dtype=np.float32)
    power = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
    starts = np.arange(0, x.size - win + 1, hop)
    ms = (power[starts + win] - power[starts]) / win
    return (10.0 * np.log10(np.maximum(ms, 1e-12))).astype(np.float32)


def _runs(mask):
    """Yield (start_idx, end_idx) for each run of True in a boolean array."""
    if mask.size == 0:
        return
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size)
    for s, e in zip(starts, ends):
        yield s, e


def _estimate_f0(x, speech_mask_frames):
    """Voiced-frame F0 in Hz by normalised autocorrelation.

    Only frames the level-based VAD already called speech are examined, and at
    most F0_MAX_FRAMES of those, sampled evenly across the lecture so the
    distribution reflects the whole thing rather than its first ten minutes.
    """
    n_frames = 1 + (x.size - F0_WIN) // F0_HOP if x.size >= F0_WIN else 0
    if n_frames <= 0:
        return np.zeros(0)

    # Map the 10 ms level grid onto the 20 ms F0 grid to reuse the VAD.
    idx = np.arange(n_frames)
    level_idx = np.minimum((idx * F0_HOP) // LEVEL_HOP,
                           max(speech_mask_frames.size - 1, 0))
    voiced_candidates = idx[speech_mask_frames[level_idx]] \
        if speech_mask_frames.size else idx
    if voiced_candidates.size == 0:
        return np.zeros(0)
    if voiced_candidates.size > F0_MAX_FRAMES:
        pick = np.linspace(0, voiced_candidates.size - 1, F0_MAX_FRAMES)
        voiced_candidates = voiced_candidates[pick.astype(int)]

    nfft = 1
    while nfft < 2 * F0_WIN:
        nfft *= 2
    lag_lo = max(int(SR / F0_MAX_HZ), 1)
    lag_hi = min(int(SR / F0_MIN_HZ), F0_WIN - 1)
    window = np.hanning(F0_WIN).astype(np.float32)

    out = []
    for block in np.array_split(voiced_candidates,
                                max(1, voiced_candidates.size // 4096)):
        starts = block * F0_HOP
        frames = np.stack([x[s:s + F0_WIN] for s in starts]).astype(np.float32)
        frames -= frames.mean(axis=1, keepdims=True)
        frames *= window
        spec = np.fft.rfft(frames, n=nfft, axis=1)
        acf = np.fft.irfft(np.abs(spec) ** 2, n=nfft, axis=1)[:, :lag_hi + 1]
        zero = acf[:, :1]
        acf = acf / np.maximum(zero, 1e-12)

        seg = acf[:, lag_lo:lag_hi + 1]
        best = np.argmax(seg, axis=1)
        peak = seg[np.arange(seg.shape[0]), best]
        lag = best + lag_lo

        # Parabolic interpolation around the peak, for sub-sample accuracy --
        # without it F0 quantises to the lag grid and fakes a low variance.
        lo = acf[np.arange(acf.shape[0]), np.maximum(lag - 1, 0)]
        hi = acf[np.arange(acf.shape[0]), np.minimum(lag + 1, lag_hi)]
        denom = (lo - 2.0 * peak + hi)
        shift = np.where(np.abs(denom) > 1e-9, 0.5 * (lo - hi) / denom, 0.0)
        shift = np.clip(shift, -1.0, 1.0)

        f0 = SR / np.maximum(lag + shift, 1e-6)
        keep = (peak >= F0_VOICED_R) & (f0 >= F0_MIN_HZ) & (f0 <= F0_MAX_HZ)
        out.append(f0[keep])
    return np.concatenate(out) if out else np.zeros(0)


def measure(pcm, loudness, duration_s):
    """(metrics, levels) for the signal tier.

    `levels` is the frame-level analysis -- dB per frame, the room-tone floor,
    the speech mask -- handed on to speech_metrics so that the transcript-based
    metrics can ask "was there audio where this word is supposed to be" without
    decoding the file a second time.
    """
    m = dict(loudness)          # loudness_lufs, loudness_range_lu, true_peak_dbtp
    levels = {}
    if pcm.size == 0:
        return m, levels

    m["clipped_pct"] = float(np.mean(np.abs(pcm) >= 0.999) * 100.0)

    db = _frame_rms_db(pcm, LEVEL_WIN, LEVEL_HOP)
    if db.size == 0:
        return m, levels
    frame_s = LEVEL_HOP / SR

    audible = db[db > DIGITAL_SILENCE_DB]
    if audible.size < 10:
        # Nothing but digital silence: report it as such rather than inventing
        # a floor, and let the not_silent gate fail on speech_pct.
        m["noise_floor_dbfs"] = float(np.max(db)) if db.size else -120.0
        m["snr_db"] = 0.0
        m["speech_pct"] = 0.0
        return m, levels

    floor_db = float(np.percentile(audible, 10.0))
    vad = db > (floor_db + VAD_MARGIN_DB)
    speech_db = db[vad]
    m["noise_floor_dbfs"] = floor_db
    m["snr_db"] = float(np.median(speech_db) - floor_db) if speech_db.size else 0.0
    m["speech_pct"] = float(vad.mean() * 100.0)
    m["speech_seconds"] = float(vad.sum() * frame_s)
    m["energy_variety_db"] = float(np.std(speech_db)) if speech_db.size > 10 else 0.0
    levels = {"db": db, "floor_db": floor_db, "vad": vad, "frame_s": frame_s}

    # Level stability: how far the median speech level wanders minute to
    # minute. Integrated loudness cannot see this -- it averages away exactly
    # the variation in question.
    frames_per_min = int(60.0 / frame_s)
    per_minute = []
    for i in range(0, max(db.size - frames_per_min, 0), frames_per_min):
        w_db, w_vad = db[i:i + frames_per_min], vad[i:i + frames_per_min]
        if w_vad.sum() > frames_per_min * 0.05:
            per_minute.append(float(np.median(w_db[w_vad])))
    if len(per_minute) >= 3:
        m["level_stability_db"] = float(np.std(per_minute))

    # Prosody.
    f0 = _estimate_f0(pcm, vad)
    if f0.size >= 200:
        # Trim the tails before taking a spread: normalised autocorrelation
        # halves or doubles on a minority of frames, and an octave error is
        # 12 semitones of pure noise straight into the number that is supposed
        # to measure expressiveness.
        lo, hi = np.percentile(f0, [5.0, 95.0])
        core = f0[(f0 >= lo) & (f0 <= hi)]
        if core.size >= 100:
            semitones = 12.0 * np.log2(core / np.median(core))
            m["pitch_variety_st"] = float(np.std(semitones))
            m["median_f0_hz"] = float(np.median(core))
            m["voiced_frames"] = int(f0.size)
    return m, levels
