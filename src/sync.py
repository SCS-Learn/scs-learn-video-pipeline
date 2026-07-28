import subprocess


def get_duration(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def align_screen_to_camera(camera_path, screen_path, out_screen_path, min_offset=0.5):
    camera_duration = get_duration(camera_path)
    screen_duration = get_duration(screen_path)
    offset = screen_duration - camera_duration

    if offset <= min_offset:
        return screen_path

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            screen_path,
            "-ss",
            str(offset),
            "-c",
            "copy",
            out_screen_path,
        ],
        check=True,
    )
    return out_screen_path


def main():
    align_screen_to_camera(
        camera_path="data/15210-lecture12/camera.mp4",
        screen_path="data/15210-lecture12/screen.mp4",
        out_screen_path="data/15210-lecture12/screen_sync.mp4",
    )


if __name__ == "__main__":
    main()
