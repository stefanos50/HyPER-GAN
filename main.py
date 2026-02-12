import os
import argparse
from pathlib import Path
import numpy as np
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import vgg16
from PIL import Image
import faiss
from tqdm import tqdm

# -------------------------------
# Dataset
# -------------------------------

class PairedDataset(Dataset):
    """Rendered -> Enhanced pairs"""
    def __init__(self, rendered_dir, enhanced_dir, size=512):
        rendered_dir = Path(rendered_dir)
        enhanced_dir = Path(enhanced_dir)
        #rendered_map = {p.stem: p for p in rendered_dir.glob("*.png")}
        rendered_map = {
            p.stem: p
            for p in rendered_dir.glob("*.png")
            if not p.name.startswith("__")
        }
        pairs = []
        for p in chain(enhanced_dir.glob("*.jpg"),enhanced_dir.glob("*.png")):
            if p.stem in rendered_map:
                pairs.append((rendered_map[p.stem], p))
        if len(pairs) == 0:
            raise RuntimeError("No matching rendered/enhanced pairs found.")
        self.rendered, self.enhanced = zip(*sorted(pairs))
        self.tf = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        print(f"[Dataset] Loaded {len(self.rendered)} pairs")

    def __len__(self):
        return len(self.rendered)

    def __getitem__(self, i):
        x = self.tf(Image.open(self.rendered[i]).convert("RGB"))
        y = self.tf(Image.open(self.enhanced[i]).convert("RGB"))
        return x, y

class RealDataset(Dataset):
    """Real-world images"""
    def __init__(self, real_dir, size=512):
        self.paths = sorted(Path(real_dir).glob("*"))
        self.tf = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))

# -------------------------------
# Models
# -------------------------------

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
    """Lightweight U-Net style generator"""
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch*2, 4, 2, 1), nn.InstanceNorm2d(base_ch*2), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1), nn.InstanceNorm2d(base_ch*4), nn.ReLU(True))
        self.middle = nn.Sequential(*[ResBlock(base_ch*4) for _ in range(4)])
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, 2, 1), nn.InstanceNorm2d(base_ch*2), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(base_ch*4, base_ch, 4, 2, 1), nn.InstanceNorm2d(base_ch), nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(base_ch*2, in_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m = self.middle(e3)
        d3 = self.dec3(m)
        d2 = self.dec2(torch.cat([d3, e2], 1))
        out = self.dec1(torch.cat([d2, e1], 1))
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 1)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------------
# VGG Feature Extractor
# -------------------------------

class VGGFeature(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features
        self.slice = nn.Sequential(*list(vgg.children())[:23])
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, x):
        f = self.slice(x)
        f = f.mean(dim=(2,3))
        return F.normalize(f, dim=1)

# -------------------------------
# Patch utilities
# -------------------------------

def split_into_crops(img, crop=196, stride=196):
    B, C, H, W = img.shape
    patches = []
    for b in range(B):
        for r in range(0, H - crop + 1, stride):
            for c in range(0, W - crop + 1, stride):
                patches.append(img[b:b+1, :, r:r+crop, c:c+crop])
    return torch.cat(patches, 0)

@torch.no_grad()
def build_faiss_index(real_loader, vgg, device, max_imgs=None):
    feats, crops = [], []
    for i, img in enumerate(real_loader):
        if max_imgs is not None and i >= max_imgs:
            break
        img = img.to(device)
        c = split_into_crops(img)
        f = vgg(c)
        feats.append(f.cpu().numpy())
        crops.append(c.cpu())
    feats = np.concatenate(feats, 0).astype(np.float32)
    crops = torch.cat(crops, 0)
    index = faiss.IndexFlatL2(feats.shape[1])
    index.add(feats)
    return index, crops

def sample_pairwise_patches(fake, enhanced, n=64, crop=196, stride=196):
    fake_crops = split_into_crops(fake, crop, stride)
    real_crops = split_into_crops(enhanced, crop, stride)
    idx = np.random.permutation(len(fake_crops))[:n]
    return fake_crops[idx], real_crops[idx]

def sample_matched_patches(fake, real_crops, index, vgg, device, n=64):
    fake_crops = split_into_crops(fake)
    with torch.no_grad():
        f = vgg(fake_crops).cpu().numpy().astype(np.float32)
    _, I = index.search(f, 1)
    idx = np.random.permutation(len(I))[:n]
    return fake_crops[idx].to(device), real_crops[I[idx,0]].to(device)

# -------------------------------
# Debug visualization
# -------------------------------

def save_patch_pairs(fake_p, real_p, out_dir, step, max_pairs=8):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    if isinstance(fake_p, list):
        fake_p = torch.cat(fake_p, 0)
    if isinstance(real_p, list):
        real_p = torch.cat(real_p, 0)

    fake_p = fake_p.detach().cpu()
    real_p = real_p.detach().cpu()

    fake_p = (fake_p * 0.5 + 0.5).clamp(0, 1)
    real_p = (real_p * 0.5 + 0.5).clamp(0, 1)

    n_pairs = min(max_pairs, fake_p.shape[0])
    for i in range(n_pairs):
        pair = torch.cat([fake_p[i], real_p[i]], dim=2)
        T.ToPILImage()(pair).save(out_dir / f"step_{step:06d}_pair_{i}.png")

# -------------------------------
# Training
# -------------------------------

def gan_loss(pred, real):
    tgt = torch.ones_like(pred) if real else torch.zeros_like(pred)
    return F.mse_loss(pred, tgt)

def train(args):
    device = torch.device("cuda")
    paired_loader = DataLoader(PairedDataset(args.rendered, args.enhanced), batch_size=1, shuffle=True)
    real_loader = DataLoader(RealDataset(args.real), batch_size=1, shuffle=True)

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)
    vgg = VGGFeature().to(device).eval()

    opt_G = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5,0.999))
    opt_D = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5,0.999))

    print("Building FAISS index from real images...")
    index, real_crops = build_faiss_index(real_loader, vgg, device)

    step = 0
    for epoch in range(args.epochs):
        for x, y in tqdm(paired_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            fake = G(x)

            # 1) Pairwise GT-enhanced supervision
            fake_p_gt, real_p_gt = sample_pairwise_patches(fake, y)

            # 2) Matched real patches for realism enforcement
            fake_p_real, real_p_real = sample_matched_patches(fake, real_crops, index, vgg, device)

            # Combine for discriminator
            fake_p = torch.cat([fake_p_gt.to(device), fake_p_real], 0)
            real_p = torch.cat([real_p_gt.to(device), real_p_real], 0)

            if step % 1000 == 0:
                save_patch_pairs(fake_p, real_p, "debug_patch_matches", step)

            # --- Train Discriminator ---
            opt_D.zero_grad()
            loss_D = (
                gan_loss(D(fake_p.detach()), False) +
                gan_loss(D(real_p), True)
            )
            loss_D.backward()
            opt_D.step()

            # --- Train Generator ---
            opt_G.zero_grad()
            loss_G = (
                gan_loss(D(fake_p), True) + 
                10.0 * F.l1_loss(fake, y)
            )
            loss_G.backward()
            opt_G.step()

            step += 1
        torch.save(G.state_dict(), f"generator_epoch_{epoch}.pth")

# -------------------------------
# Inference
# -------------------------------

@torch.no_grad()
def inference(args):
    device = torch.device("cuda")
    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
    G.eval()

    tf = T.Compose([
        T.Resize((544,960)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)

    for p in Path(args.input).glob("*"):
        img = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        out = G(img)[0]
        out = (out * 0.5 + 0.5).clamp(0,1)
        T.ToPILImage()(out.cpu()).save(out_dir / p.name.replace(".png",".jpg"))

# -------------------------------
# Main #python preprocessing.py --mode train --rendered ./GTA/train --enhanced ./FULL/EPE_GTA --real ./Cityscapes 
# -------------------------------

#python main.py --mode infer --input ./test_images --ckpt ./pretrained_models/gta2cs.pth --out ./output
#python main.py --mode train --rendered <path-to-rendered-dir> --enhanced <path-to-enhanced-dir> --real <path-to-real-dir>

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","infer"], required=True)
    parser.add_argument("--rendered", type=str)
    parser.add_argument("--enhanced", type=str)
    parser.add_argument("--real", type=str)
    parser.add_argument("--epochs", type=int, default=21)
    parser.add_argument("--input", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        inference(args)
