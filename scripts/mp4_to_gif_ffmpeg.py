import os
import argparse
import subprocess
import tempfile

import imageio_ffmpeg


def mp4_to_gif_ffmpeg(input_path: str, output_path: str, fps: int = 15, width: int | None = 640):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        palette = os.path.join(td, 'palette.png')
        scale_filter = f"scale={width}:-1:flags=lanczos" if width else "scale=iw:ih"
        # 1) Generate palette for better colors
        subprocess.run([
            ffmpeg, '-y', '-i', input_path,
            '-vf', f"fps={fps},{scale_filter},palettegen",
            palette
        ], check=True)
        # 2) Use palette to create optimized gif
        subprocess.run([
            ffmpeg, '-y', '-i', input_path, '-i', palette,
            '-lavfi', f"fps={fps},{scale_filter} [x]; [x][1:v] paletteuse",
            output_path
        ], check=True)


def main():
    parser = argparse.ArgumentParser(description='Convert MP4 to GIF using bundled ffmpeg')
    parser.add_argument('--input', required=True, help='Input MP4 path')
    parser.add_argument('--output', required=True, help='Output GIF path')
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--width', type=int, default=640)
    args = parser.parse_args()

    mp4_to_gif_ffmpeg(args.input, args.output, fps=args.fps, width=args.width)


if __name__ == '__main__':
    main()


