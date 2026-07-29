import subprocess
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
    d = get_duration("data/15210-lecture12/camera_muted.mp4")
    composite_pip(
        screen_path="data/15210-lecture12/screen_with_cards.mp4",
        camera_path="data/15210-lecture12/camera_muted.mp4",
        out_path="data/15210-lecture12/15210-lecture12.mp4",
        duration=d,
    )


if __name__ == "__main__":
    main()
