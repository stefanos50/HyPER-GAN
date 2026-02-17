import carla
import pygame
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Model
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
# Args
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


# =========================================================
# Load model
# =========================================================

def load_model(path):
    model = UNetGenerator().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# =========================================================
# Main
# =========================================================

def main():

    args = parse_args()
    model = load_model(args.ckpt)

    tf = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])

    # ==============================
    # Pygame window (DOUBLE WIDTH)
    # ==============================
    pygame.init()
    display = pygame.display.set_mode((args.width * 2, args.height))
    pygame.display.set_caption("CARLA Original | Enhanced")

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # ==============================
    # Sync mode
    # ==============================
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # ==============================
    # Spawn vehicle
    # ==============================
    vehicle_bp = np.random.choice(bp_lib.filter("vehicle.*"))
    spawn = np.random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn)

    # ==============================
    # Camera
    # ==============================
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(args.width))
    cam_bp.set_attribute("image_size_y", str(args.height))

    cam_tf = carla.Transform(carla.Location(x=-7, z=1.4))
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

    image_queue = []

    def callback(image):
        image_queue.append(image)

    camera.listen(callback)

    control = carla.VehicleControl()

    print("WASD to drive | ESC to quit")

    try:
        while True:

            world.tick()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            keys = pygame.key.get_pressed()

            control.throttle = 0
            control.brake = 0
            control.steer = 0

            if keys[pygame.K_w]:
                control.throttle = 0.6
            if keys[pygame.K_s]:
                control.brake = 1.0
            if keys[pygame.K_a]:
                control.steer = -0.5
            if keys[pygame.K_d]:
                control.steer = 0.5
            if keys[pygame.K_ESCAPE]:
                return

            vehicle.apply_control(control)

            if not image_queue:
                continue

            image = image_queue.pop(0)

            # ==============================
            # Convert BGRA → RGB
            # ==============================
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            rgb = array[:, :, :3][:, :, ::-1]

            # ==============================
            # ORIGINAL SURFACE
            # ==============================
            orig_surface = pygame.surfarray.make_surface(
                rgb.swapaxes(0, 1)
            )

            # ==============================
            # MODEL INFERENCE
            # ==============================
            img = tf(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                out = model(img)[0]
                out = (out * 0.5 + 0.5).clamp(0, 1)

            out_np = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # ==============================
            # ENHANCED SURFACE
            # ==============================
            enh_surface = pygame.surfarray.make_surface(
                out_np.swapaxes(0, 1)
            )

            # ==============================
            # DISPLAY SIDE-BY-SIDE
            # ==============================
            display.blit(orig_surface, (0, 0))
            display.blit(enh_surface, (args.width, 0))

            pygame.display.flip()

    finally:

        print("Cleaning up")

        camera.stop()
        camera.destroy()
        vehicle.destroy()

        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        pygame.quit()


if __name__ == "__main__":
    main()
