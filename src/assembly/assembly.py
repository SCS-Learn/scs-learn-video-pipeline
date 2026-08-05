import os
import subprocess
from src.paths import LecturePaths, lecture_parser
from src.sync import get_duration


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


def main():
    parser = lecture_parser("Composite the final picture-in-picture video.")
    parser.add_argument("--allow-unanonymized", action="store_true",
                        help="Proceed even if no face-anonymized camera exists")
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
    args = parser.parse_args()
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

    if not args.skip_pip:
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
        print(f"[assembly] wrote {out}")

    if args.camera_only or args.skip_pip:
        out2 = camera_only(camera, p.final_camera_only, d)
        print(f"[assembly] wrote {out2} (camera only, no screen)")


if __name__ == "__main__":
    main()
