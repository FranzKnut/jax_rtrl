import os
import numpy as np
import cv2
from PIL import Image

DATA_FOLDER = "data/first"


def make_video(frames, fps=30, output_file="video.mp4"):
    shape = frames.shape[2], frames.shape[1]
    print("Writing Video to", output_file)
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"avc1"), fps, shape)
    for f in frames:
        out.write(f)
    out.release()


def make_gif(frames, fps=30, output_file="artifacts/videos/video.gif", scale=0.5, downsample=5):
    new_shape = [int(s * scale) for s in reversed(frames.shape[1:-1])]

    def transform(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return Image.fromarray(img).resize(new_shape)

    imgs = [transform(img) for img in frames]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    print("Writing Video to", output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    imgs[0].save(output_file, save_all=True, append_images=imgs[1::downsample], duration=1000 // (fps * downsample), loop=0)


print("Reading data ...")
frames = {}
for f in os.listdir(DATA_FOLDER):
    data = np.load(os.path.join(DATA_FOLDER, f))
    img = data["vision"][..., :-1]
    key = data["ts"][0]
    frames[key] = img

frames = [frames[k] for k in sorted(frames)]
frames = np.concatenate(frames).astype(np.uint8)
# make_video(frames)
make_gif(frames, output_file=f"artifacts/videos/{os.path.basename(DATA_FOLDER)}.gif")
