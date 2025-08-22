import os
import argparse
from moviepy.editor import VideoFileClip


def mp4_to_gif(input_path: str, output_path: str, fps: int = 15, width: int | None = 640):
    clip = VideoFileClip(input_path)
    if width is not None:
        clip = clip.resize(width=width)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    clip.write_gif(output_path, fps=fps)


def main():
    parser = argparse.ArgumentParser(description='Convert MP4 to GIF')
    parser.add_argument('--input', required=True, help='Input MP4 path')
    parser.add_argument('--output', required=True, help='Output GIF path')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second for GIF')
    parser.add_argument('--width', type=int, default=640, help='Width to scale GIF to (keep aspect)')
    args = parser.parse_args()

    mp4_to_gif(args.input, args.output, fps=args.fps, width=args.width)


if __name__ == '__main__':
    main()


