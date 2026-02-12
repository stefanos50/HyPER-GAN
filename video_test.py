import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Video Processing Script")

    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Path to input video file (.mkv)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./resvid",
        help="Directory to save results"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint (.pth)"
    )

    return parser.parse_args()


DEVICE = "cuda"   # or "cpu"


# =========================================================
# Model definition
# =========================================================

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.InstanceNorm2d(c),
            nn.ReLU(True),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.InstanceNorm2d(c)
        )

    def forward(self, x):
        return x + self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1),
            nn.ReLU(True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),
            nn.InstanceNorm2d(base_ch * 2),
            nn.ReLU(True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),
            nn.InstanceNorm2d(base_ch * 4),
            nn.ReLU(True)
        )

        self.middle = nn.Sequential(
            *[ResBlock(base_ch * 4) for _ in range(4)]
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1),
            nn.InstanceNorm2d(base_ch * 2),
            nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch, 4, 2, 1),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, in_ch, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        m = self.middle(e3)

        d3 = self.dec3(m)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        out = self.dec1(torch.cat([d2, e1], dim=1))
        return out


# =========================================================
# Video inference
# =========================================================

@torch.no_grad()
def enhance_video():
    device = torch.device(DEVICE)

    # Load model
    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    G.eval()

    # Open video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(INPUT_VIDEO).stem + "_enhanced.mp4")

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    tf = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])

    print(f"[INFO] Processing {n_frames} frames...")
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB → Tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tf(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)

        # Inference
        out = G(img)[0]
        out = (out * 0.5 + 0.5).clamp(0, 1)

        # Tensor → uint8 → BGR
        out_np = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

        writer.write(out_bgr)

        frame_id += 1
        if frame_id % 50 == 0:
            print(f"  processed {frame_id}/{n_frames} frames")

    cap.release()
    writer.release()
    print(f"[✓] Enhanced video saved to: {out_path}")


# =========================================================
# Run
# =========================================================

#python video_test.py --input_video ./test_videos/001.mp4 --output_dir ./ --ckpt ./pretrained_models/gta2cs.pth
if __name__ == "__main__":
    args = parse_args()

    INPUT_VIDEO = args.input_video
    OUTPUT_DIR  = args.output_dir
    CKPT_PATH   = args.ckpt

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Input video:", INPUT_VIDEO)
    print("Output dir:", OUTPUT_DIR)
    print("Checkpoint:", CKPT_PATH)

    enhance_video()
