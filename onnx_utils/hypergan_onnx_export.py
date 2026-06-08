import argparse
import torch
import torch.nn as nn
import torch.onnx


# ============================================================
# Model Definitions
# ============================================================

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
        d2 = self.dec2(torch.cat([d3, e2], 1))

        out = self.dec1(torch.cat([d2, e1], 1))

        return out


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export REGEN Generator from PyTorch to ONNX"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to .pth model"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output ONNX file"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Input height"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Input width"
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=9,
        help="ONNX opset version"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.input}")

    model = UNetGenerator()

    checkpoint = torch.load(args.input, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        args.height,
        args.width
    )

    print(
        f"Exporting ONNX ({args.width}x{args.height}) -> {args.output}"
    )

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False
    )

    print(f"Successfully exported: {args.output}")


if __name__ == "__main__":
    main()