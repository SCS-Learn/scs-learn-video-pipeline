"""Finish the deliverable: take the brand layout render and add the sound.

    python -m src.assembly.assembly --lecture-dir data/15210-lecture12

The picture is decided by src/assembly/layout.py, which composites the two SCS
Open Courseware scenes from assets/brand/plates. This stage does NOT re-frame
anything -- it stream-copies that video, mixes the question-card sting into the
silences the audio pass left, and optionally writes the camera-only cut.

Splitting it that way keeps the plate geometry in exactly one place. The older
behaviour -- compositing a 480px corner picture-in-picture over the screen right
here -- is still in this file as `composite_pip`, reachable with --legacy-pip,
because it is what every video published before the brand assets arrived looks
like. It is not a fallback: if the layout render is missing, this stage stops
rather than quietly publishing the wrong look under the same filename.
"""

import json
import os
import re
import subprocess
import tempfile

from src.assembly import brand
from src.paths import LecturePaths, lecture_parser
from src.sync import get_duration


# --- the pre-brand layout, kept for reproducing older deliverables ----------
# Where the picture-in-picture sits. Commit b8b4089 moved it to top-right; the
# position is a flag now so that choice is visible rather than baked in.
PIP_POSITIONS = {
    "bottom-right": "W-w-{m}:H-h-{m}",
    "top-right": "W-w-{m}:{m}",
    "bottom-left": "{m}:H-h-{m}",
    "top-left": "{m}:{m}",
}


def composite_pip(
    screen_path, camera_path, out_path, duration, pip_width=480, margin=30,
    position="bottom-right"
):
    pip_height = int(pip_width * 9 / 16)
    overlay_xy = PIP_POSITIONS[position].format(m=margin)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        screen_path,
        "-i",
        camera_path,
        "-filter_complex",
        f"[0:v]scale=1920:1080[main];"
        f"[1:v]scale={pip_width}:{pip_height}[pip];"
        f"[main][pip]overlay={overlay_xy}[outv]",
        "-map",
        "[outv]",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-t",
        str(duration),
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def finish_layout(layout_path, out_path):
    """The layout render, remuxed to the deliverable's name.

    Video is stream-copied -- re-encoding a finished 90-minute libx264 pass to
    change nothing but the filename would cost an hour and a generation of
    quality. +faststart is the one thing worth a pass over the file: without it
    the moov atom sits at the end and the video will not start playing until it
    has downloaded.

    No -t: the layout render already covers exactly the span it was asked for,
    and trimming it to the camera's duration instead would cut the tail off any
    lecture whose screen outlasts the camera.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", layout_path,
         "-c", "copy", "-movflags", "+faststart", out_path],
        check=True)
    return out_path




# --- the opening card ------------------------------------------------------
INTRO_SECONDS = 5.0
INTRO_FADE = 0.6
# Where the intro's music should sit relative to the LECTURE's own loudness.
#
# Measured: the supplied sting `assets/themes/fun/intro-card-fun.mp4` is
# -15.5 LUFS and lecture 12's finished audio is -20.9, so at unity the intro
# arrives 5.4 dB hotter than everything that follows it. That is the "blaring"
# -- not that the sting is loud in absolute terms, but that it is louder than
# the programme, so a viewer sets their volume to the intro and then cannot
# hear the lecture.
#
# So the gain is DERIVED, not fixed: measure the body, measure the sting, and
# put the sting this far under the body. -2 dB is just perceptibly quieter,
# which is what an opening title wants -- it should not be the loudest thing
# in the video. A fixed gain would be wrong on the next lecture, because the
# body's loudness varies by a couple of dB across the corpus.
INTRO_UNDER_DB = -2.0
INTRO_CLIP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets", "themes", "fun", "intro-card-fun.mp4")


def measure_lufs(path):
    """Integrated loudness in LUFS, or None if it cannot be measured.

    ffmpeg's loudnorm in analysis mode, which prints a JSON blob to stderr.
    One pass over the audio only -- no video is decoded -- so an 80-minute
    lecture costs seconds, not minutes.
    """
    out = subprocess.run(
        ["ffmpeg", "-v", "info", "-nostats", "-i", path, "-vn",
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    m = re.search(r"\{[^{}]*input_i[^{}]*\}", out, re.S)
    if not m:
        return None
    try:
        return float(json.loads(m.group(0))["input_i"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def prepend_intro(video_path, meta, out_path, seconds=INTRO_SECONDS,
                  fade=INTRO_FADE, clip=None, under_db=INTRO_UNDER_DB,
                  silent=False):
    """Put a branded title card, with the theme sting under it, on the front.

    Same join mechanics as the end card -- match the body's codec, pixel
    format, frame rate and TIMESCALE, then concat -- for the same reason: the
    concat demuxer stream-copies into a container that keeps the first input's
    timescale, and here the intro IS the first input, so getting it wrong
    would reinterpret every packet in the lecture behind it.
    """
    v = _probe_stream(video_path, "v")
    a = _probe_stream(video_path, "a")
    if not v:
        raise SystemExit(f"{video_path} has no video stream to match")

    work = tempfile.mkdtemp(prefix="intro-")
    png = os.path.join(work, "title.png")
    brand.build_title_card(meta, png)

    gain_db = None
    clip = clip or INTRO_CLIP
    use_sting = bool(a) and not silent and os.path.exists(clip)
    if use_sting:
        body_i, sting_i = measure_lufs(video_path), measure_lufs(clip)
        if body_i is None or sting_i is None:
            print("[assembly] could not measure loudness; intro will be silent "
                  "rather than guessed at")
            use_sting = False
        else:
            gain_db = (body_i + under_db) - sting_i
            print(f"[assembly] intro sting {sting_i:.1f} LUFS against a body of "
                  f"{body_i:.1f} LUFS -> {gain_db:+.1f} dB to sit "
                  f"{abs(under_db):.0f} dB under")

    fps = v.get("r_frame_rate") or "25/1"
    tb = (v.get("time_base") or "1/12800").split("/")
    timescale = tb[1] if len(tb) == 2 and tb[1].isdigit() else "12800"
    intro = os.path.join(work, "intro.mp4")

    cmd = ["ffmpeg", "-y", "-v", "error",
           "-loop", "1", "-framerate", fps, "-t", f"{seconds:g}", "-i", png]
    if use_sting:
        cmd += ["-i", clip]
    elif a:
        cmd += ["-f", "lavfi", "-t", f"{seconds:g}",
                "-i", f"anullsrc=r={a.get('sample_rate', 48000)}:"
                      f"cl={a.get('channels', 2)}c"]
    vf = (f"scale={v['width']}:{v['height']},setsar=1,"
          f"format={v.get('pix_fmt', 'yuv420p')},"
          f"fade=t=in:st=0:d={fade:g},"
          f"fade=t=out:st={max(0.0, seconds - fade):g}:d={fade:g}")
    cmd += ["-vf", vf]
    if a:
        af = (f"volume={gain_db:.2f}dB," if gain_db is not None else "")
        # Faded at both ends regardless of level: a sting that stops dead is
        # a click, and one that runs into the first word of the lecture is
        # the same complaint in a different place.
        af += (f"afade=t=in:st=0:d={fade:g},"
               f"afade=t=out:st={max(0.0, seconds - fade):g}:d={fade:g},"
               f"aresample={a.get('sample_rate', 48000)}")
        cmd += ["-af", af, "-c:a", "aac", "-b:a", "192k",
                "-ar", str(a.get("sample_rate", 48000)),
                "-ac", str(a.get("channels", 2))]
    cmd += ["-t", f"{seconds:g}", "-c:v", "libx264",
            "-profile:v", v.get("profile", "high").lower(),
            "-pix_fmt", v.get("pix_fmt", "yuv420p"), "-crf", "18",
            "-preset", "veryfast", "-r", fps,
            "-video_track_timescale", timescale, intro]
    subprocess.run(cmd, check=True)

    listing = os.path.join(work, "concat.txt")
    with open(listing, "w") as f:
        f.write(f"file '{os.path.abspath(intro)}'\n")
        f.write(f"file '{os.path.abspath(video_path)}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", listing, "-c", "copy", "-movflags", "+faststart", out_path],
        check=True)
    return out_path


# --- the closing card ------------------------------------------------------
END_CARD_SECONDS = 5.0
END_CARD_FADE = 0.5


def append_end_card(video_path, meta, out_path, seconds=END_CARD_SECONDS,
                    fade=END_CARD_FADE, message=None):
    """Put a thank-you / subscribe card on the end of a finished video.

    The card has to be encoded to match the body EXACTLY -- codec, profile,
    pixel format, resolution, frame rate, timebase, and an audio track with
    the same rate and channel count -- because the join is a concat demuxer
    stream copy. Anything that disagrees either fails outright or, worse,
    produces a file whose second half will not decode in some players. So the
    parameters are read off the body with ffprobe rather than assumed.

    Silent audio, not "no audio": a concat of one stream with sound and one
    without drops the audio from the join onward in most players.
    """
    v = _probe_stream(video_path, "v")
    a = _probe_stream(video_path, "a")
    if not v:
        raise SystemExit(f"{video_path} has no video stream to match")

    work = tempfile.mkdtemp(prefix="endcard-")
    png = os.path.join(work, "card.png")
    brand.build_end_card(meta, png, **({"message": message} if message else {}))

    fps = v.get("r_frame_rate") or "25/1"
    card = os.path.join(work, "card.mp4")
    # Fade in from the field colour so the cut to the card is not a flash.
    vf = (f"scale={v['width']}:{v['height']},setsar=1,format={v.get('pix_fmt', 'yuv420p')},"
          f"fade=t=in:st=0:d={fade:g}")
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-loop", "1", "-framerate", fps, "-t", f"{seconds:g}", "-i", png]
    if a:
        cmd += ["-f", "lavfi", "-t", f"{seconds:g}",
                "-i", f"anullsrc=r={a.get('sample_rate', 48000)}:"
                      f"cl={a.get('channels', 2)}c"]
    # The timescale must match the BODY's, not be some reasonable constant.
    # The concat demuxer stream-copies packets into a container that keeps the
    # first input's timescale, so a card written at 90000 against a body at
    # 12800 has every packet duration reinterpreted: the audio came out right
    # and the video reported 456s for a 65s file. Nothing errors -- the file
    # plays, and then sits on the last frame for six minutes.
    tb = (v.get("time_base") or "1/12800").split("/")
    timescale = tb[1] if len(tb) == 2 and tb[1].isdigit() else "12800"
    cmd += ["-vf", vf, "-c:v", "libx264",
            "-profile:v", v.get("profile", "high").lower(),
            "-pix_fmt", v.get("pix_fmt", "yuv420p"), "-crf", "18",
            "-preset", "veryfast", "-r", fps,
            "-video_track_timescale", timescale]
    if a:
        cmd += ["-c:a", "aac", "-b:a", "192k",
                "-ar", str(a.get("sample_rate", 48000)),
                "-ac", str(a.get("channels", 2)), "-shortest"]
    cmd += [card]
    subprocess.run(cmd, check=True)

    listing = os.path.join(work, "concat.txt")
    with open(listing, "w") as f:
        f.write(f"file '{os.path.abspath(video_path)}'\n")
        f.write(f"file '{os.path.abspath(card)}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", listing, "-c", "copy", "-movflags", "+faststart", out_path],
        check=True)
    return out_path


def _probe_stream(path, kind):
    """First stream of `kind` as a dict, or None."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", kind, "-show_streams",
         "-of", "json", path], capture_output=True, text=True).stdout
    try:
        streams = json.loads(out).get("streams") or []
    except json.JSONDecodeError:
        return None
    return streams[0] if streams else None


def camera_only(camera_path, out_path, duration, width=1920, height=1080,
                crf=18):
    """Camera-only deliverable -- no screen composited in.

    Same anonymized camera and muted audio as the PiP version, scaled to the
    main deliverable's frame size so the two sit side by side. Upscaling 720p
    adds no detail; it only makes the frame sizes match.
    """
    cmd = [
        "ffmpeg", "-y", "-i", camera_path,
        "-vf", f"scale={width}:{height}:flags=lanczos,setsar=1",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", "-t", str(duration), out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path



# question-card-sound.mp3 is mastered at -7.7 LUFS and peaks at 0.0 dBFS --
# roughly 12 dB hotter than the lecture itself, which measures -19.5 LUFS. At
# unity it lands ~15 dB above the instructor's speech, and it arrives during a
# deliberately silent stretch, so it reads as a jolt. 0.22 puts it at about
# -20.5 LUFS: just under the speech it interrupts. 0.25 is exact parity.
CARD_SOUND_GAIN = 0.22


def mix_card_sound(video_path, manifest_path, sound_path, out_path,
                   gain=CARD_SOUND_GAIN, max_cards=64):
    """Mix the card sting into the finished video at every card span.

    The screen carries no audio, so the deliverable's soundtrack is the camera's
    muted track. During a student question that track is *silent by design* --
    the audio pass mutes it -- so a card currently plays over nothing. This
    fills that silence without touching the instructor's speech.

    Video is stream-copied; only the audio is re-encoded.

    Returns the path actually written: unchanged when there is nothing to mix.
    """
    if not (os.path.exists(manifest_path) and os.path.exists(sound_path)):
        return video_path
    with open(manifest_path) as f:
        cards = json.load(f)
    if not cards:
        return video_path
    if len(cards) > max_cards:
        # One amix input per card; thousands would build an unusable graph.
        print(f"[assembly] {len(cards)} cards, mixing the first {max_cards} "
              f"only -- raise max_cards if this is real")
        cards = cards[:max_cards]

    n = len(cards)
    parts = [f"[1:a]volume={gain},asplit={n}" +
             "".join(f"[c{i}]" for i in range(n)) + ";"]
    for i, c in enumerate(cards):
        # adelay wants milliseconds per channel; all= applies it to both.
        parts.append(f"[c{i}]adelay={int(c['start'] * 1000)}:all=1[d{i}];")
    # normalize=0 keeps amix from ducking the lecture audio by 1/n every time a
    # sting plays -- the default would make the instructor quieter at each card.
    parts.append("[0:a]" + "".join(f"[d{i}]" for i in range(n)) +
                 f"amix=inputs={n + 1}:duration=first:normalize=0[aout]")
    fc = "".join(parts)

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", video_path, "-i", sound_path,
         "-filter_complex", fc, "-map", "0:v", "-map", "[aout]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart", out_path],
        check=True)
    print(f"[assembly] mixed the card sound into {n} card span(s)")
    return out_path




def _intro(args, p, out):
    """Prepend the title card, unless it was turned off. Returns `out`."""
    if args.no_intro or args.intro_seconds <= 0:
        return out
    meta = brand.lecture_meta(p.metadata)
    tmp = out + ".intro.mp4"
    try:
        prepend_intro(out, meta, tmp, seconds=args.intro_seconds,
                      clip=args.intro_clip, under_db=args.intro_under_db,
                      silent=args.intro_silent)
    except (subprocess.CalledProcessError, SystemExit) as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"[assembly] WARNING: intro not prepended ({e}). The video is "
              f"complete and unchanged; --no-intro to skip it deliberately.")
        return out
    os.replace(tmp, out)
    print(f"[assembly] prepended a {args.intro_seconds:g}s title card")
    return out



def _end_card(args, p, out):
    """Append the closing card, unless it was turned off. Returns `out`."""
    if args.no_end_card or args.end_card_seconds <= 0:
        return out
    meta = brand.lecture_meta(p.metadata)
    tmp = out + ".card.mp4"
    try:
        append_end_card(out, meta, tmp, seconds=args.end_card_seconds,
                        message=args.end_card_message)
    except (subprocess.CalledProcessError, SystemExit) as e:
        # A card is a nicety; the lecture is the deliverable. Losing the video
        # because the card would not encode is the wrong trade -- but doing it
        # silently is worse, because "no card" then looks like a flag setting.
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"[assembly] WARNING: end card not appended ({e}). The video is "
              f"complete and unchanged; re-run with --no-end-card to skip it "
              f"deliberately.")
        return out
    os.replace(tmp, out)
    print(f"[assembly] appended a {args.end_card_seconds:g}s closing card")
    return out



def main():
    parser = lecture_parser(
        "Finish the deliverable from the SCS scene layout render.")
    parser.add_argument("--allow-unanonymized", action="store_true",
                        help="Proceed even if no face-anonymized camera exists")
    parser.add_argument("--legacy-pip", action="store_true",
                        help="Composite the pre-brand 480px corner "
                             "picture-in-picture instead of using the layout "
                             "render. For reproducing videos published before "
                             "the SCS brand assets landed.")
    parser.add_argument("--pip-position", default="bottom-right",
                        choices=sorted(PIP_POSITIONS),
                        help="Corner for the picture-in-picture")
    parser.add_argument("--pip-width", type=int, default=480)
    parser.add_argument("--no-tracked", action="store_true",
                        help="Use the un-cropped camera even if a tracked crop "
                             "exists")
    parser.add_argument("--camera-only", action="store_true",
                        help="Also write <key>-camera.mp4: the anonymized camera "
                             "with muted audio and NO screen composited in")
    parser.add_argument("--skip-pip", action="store_true",
                        help="Write only the camera-only file, not the PiP one")
    parser.add_argument("--no-intro", action="store_true",
                        help="Do not put a title card on the front")
    parser.add_argument("--intro-seconds", type=float, default=INTRO_SECONDS)
    parser.add_argument("--intro-silent", action="store_true",
                        help="Title card with no music under it")
    parser.add_argument("--intro-under-db", type=float, default=INTRO_UNDER_DB,
                        help="How far under the LECTURE's own loudness the "
                             "intro sting sits. Measured per lecture, not a "
                             "fixed gain: the supplied sting is 5.4 dB hotter "
                             "than lecture 12's finished audio, which is what "
                             "makes it blare.")
    parser.add_argument("--intro-clip", default=None,
                        help=f"Audio bed for the title card. Default "
                             f"{os.path.relpath(INTRO_CLIP, os.getcwd()) if INTRO_CLIP.startswith(os.getcwd()) else 'the fun theme sting'}")
    parser.add_argument("--no-end-card", action="store_true",
                        help="Do not append the thank-you / subscribe card")
    parser.add_argument("--end-card-seconds", type=float,
                        default=END_CARD_SECONDS)
    parser.add_argument("--end-card-message", default=None,
                        help=f"Overrides {brand.END_CARD_MESSAGE!r}")
    parser.add_argument("--layout", default=brand.DEFAULT_LAYOUT,
                        choices=sorted(brand.LAYOUTS),
                        help="Which geometry the end card should match. Use "
                             "the same value the layout render used.")
    parser.add_argument("--no-card-sound", action="store_true",
                        help="Do not mix the sting over question cards")
    parser.add_argument("--card-sound-gain", type=float, default=CARD_SOUND_GAIN,
                        help="Linear gain for the sting. The supplied file is "
                             "mastered ~12 dB hotter than the lecture.")
    args = parser.parse_args()
    brand.set_layout(args.layout)
    p = LecturePaths(args.lecture_dir)

    # Prefer the face-anonymized camera. This used to hardcode camera_muted.mp4,
    # which silently discarded a completed anonymization pass and published a
    # video still showing every student's face.
    camera = p.resolve_camera_for_assembly()
    if camera == p.camera_muted and not args.allow_unanonymized:
        raise SystemExit(
            f"refusing to assemble: {p.camera_anon} does not exist, so faces "
            f"are NOT anonymized.\n"
            f"Run:  python -m src.video.face_anon --lecture-dir {p.dir} "
            f"--input camera_muted.mp4\n"
            f"or pass --allow-unanonymized if that is genuinely intended.")
    print(f"[assembly] camera={camera}")

    d = get_duration(camera)

    if not args.skip_pip and not args.legacy_pip:
        if not os.path.exists(p.layout):
            raise SystemExit(
                f"refusing to assemble: {p.layout} does not exist, so the SCS "
                f"scenes have not been rendered.\n"
                f"Run:  python -m src.assembly.layout --lecture-dir {p.dir}\n"
                f"This stage will not fall back to the old corner "
                f"picture-in-picture on its own -- that would publish a "
                f"different-looking video under the same filename depending on "
                f"which stages happened to run. Pass --legacy-pip if you "
                f"genuinely want the pre-brand layout.")
        print(f"[assembly] picture from {p.layout} (SCS scene layout)")
        out = finish_layout(p.layout, p.final)
        if not args.no_card_sound:
            tmp = out + ".snd.mp4"
            if mix_card_sound(out, p.cards_manifest, p.card_sound, tmp,
                              gain=args.card_sound_gain) == tmp:
                os.replace(tmp, out)
            elif os.path.exists(tmp):
                os.remove(tmp)
        out = _end_card(args, p, out)
        out = _intro(args, p, out)
        print(f"[assembly] wrote {out}")

    if args.legacy_pip and not args.skip_pip:
        print("[assembly] --legacy-pip: pre-brand corner picture-in-picture")
        pip_src = camera if args.no_tracked else p.resolve_pip_camera()
        # Always say which source won. The choice depends on whether a file
        # happens to exist, so silence here meant two runs over the same lecture
        # could differ with nothing in the log to explain it.
        if pip_src != camera:
            print(f"[assembly] pip source={pip_src} (instructor-tracked crop)")
        elif args.no_tracked:
            print(f"[assembly] pip source={pip_src} (--no-tracked: uncropped)")
        else:
            print(f"[assembly] pip source={pip_src} (uncropped -- "
                  f"{os.path.basename(p.camera_tracked)} not found; run "
                  f"'python -m src.video.track_instructor --lecture-dir {p.dir}' "
                  f"for a closer picture-in-picture)")
        out = composite_pip(
            screen_path=p.screen_with_cards,
            camera_path=pip_src,
            out_path=p.final,
            duration=d,
            pip_width=args.pip_width,
            position=args.pip_position,
        )
        if not args.no_card_sound:
            tmp = out + ".snd.mp4"
            if mix_card_sound(out, p.cards_manifest, p.card_sound, tmp,
                              gain=args.card_sound_gain) == tmp:
                os.replace(tmp, out)
            elif os.path.exists(tmp):
                os.remove(tmp)
        out = _end_card(args, p, out)
        out = _intro(args, p, out)
        print(f"[assembly] wrote {out}")

    if args.camera_only or args.skip_pip:
        out2 = camera_only(camera, p.final_camera_only, d)
        print(f"[assembly] wrote {out2} (camera only, no screen)")


if __name__ == "__main__":
    main()
