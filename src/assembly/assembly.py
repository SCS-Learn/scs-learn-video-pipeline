import subprocess
from src.paths import LecturePaths, lecture_parser
from src.sync import get_duration


def composite_pip(
    screen_path, camera_path, out_path, duration, pip_width=480, margin=30
):
    pip_height = int(pip_width * 9 / 16)

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
        f"[main][pip]overlay=W-w-{margin}:{margin}[outv]",
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


def main():
    parser = lecture_parser("Composite the final picture-in-picture video.")
    parser.add_argument("--allow-unanonymized", action="store_true",
                        help="Proceed even if no face-anonymized camera exists")
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
    out = composite_pip(
        screen_path=p.screen_with_cards,
        camera_path=camera,
        out_path=p.final,
        duration=d,
    )
    print(f"[assembly] wrote {out}")


if __name__ == "__main__":
    main()
