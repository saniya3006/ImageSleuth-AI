"""Generate SYNTHETIC demonstration images for ImageSleuth AI.

Every image produced here is computer-generated. The edits are made
deliberately and their exact locations are written to ``manifest.json`` so
that a demo can show what the tools respond to. These samples are NOT
real-world photographs and results on them say nothing about real-world
forensic accuracy.

Usage (from the project root)::

    python scripts/generate_demo_images.py            # writes to assets/demo/
    python scripts/generate_demo_images.py --out some/folder --seed 7
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

WIDTH, HEIGHT = 1024, 768
LABEL = "SYNTHETIC DEMO IMAGE - computer-generated, not a real photograph"

WORDS = ["ALPHA", "DELTA", "NORTH", "GATE 7", "K-2049", "OMEGA", "SECTOR", "ZONE B", "LAB-3", "VECTOR"]


def _smooth_noise(rng: np.random.Generator, height: int, width: int, scale: int) -> np.ndarray:
    """Random value noise upsampled from a coarse grid: organic, non-repeating texture."""
    coarse = rng.random((max(2, height // scale), max(2, width // scale))).astype(np.float32)
    return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)


def make_scene(seed: int) -> np.ndarray:
    """Create a textured, non-repetitive synthetic RGB scene."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None]
    sky = np.dstack([40 + 60 * y, 70 + 80 * y, 120 + 70 * y]) * np.ones((1, WIDTH, 1), np.float32)
    texture = sum(_smooth_noise(rng, HEIGHT, WIDTH, s) * w for s, w in ((64, 40), (16, 25), (4, 12)))
    scene = np.clip(sky + texture[..., None], 0, 255).astype(np.uint8)
    ground_top = int(HEIGHT * 0.55)
    ground = np.clip(
        np.dstack([70 + texture * 1.4, 95 + texture * 1.2, 50 + texture * 0.6])[ground_top:], 0, 255
    ).astype(np.uint8)
    scene[ground_top:] = ground

    for _ in range(38):  # random shapes with unique sizes, colours and positions
        color = tuple(int(c) for c in rng.integers(30, 240, 3))
        kind = rng.integers(0, 3)
        cx, cy = int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT))
        if kind == 0:
            cv2.circle(scene, (cx, cy), int(rng.integers(8, 60)), color, -1, cv2.LINE_AA)
        elif kind == 1:
            w, h = int(rng.integers(15, 120)), int(rng.integers(15, 120))
            cv2.rectangle(scene, (cx, cy), (cx + w, cy + h), color, int(rng.choice([-1, 2, 4])))
        else:
            pts = rng.integers(-60, 60, (5, 2)) + np.array([cx, cy])
            cv2.fillPoly(scene, [pts.astype(np.int32)], color, cv2.LINE_AA)
    for _ in range(14):
        cv2.putText(scene, str(rng.choice(WORDS)), (int(rng.integers(0, WIDTH - 150)), int(rng.integers(30, HEIGHT))),
                    cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.5, 1.4)), tuple(int(c) for c in rng.integers(0, 255, 3)),
                    int(rng.integers(1, 3)), cv2.LINE_AA)
    noise = rng.normal(0, 4.0, scene.shape)  # mild sensor-like noise
    scene = np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.putText(scene, "SYNTHETIC DEMO", (WIDTH - 230, HEIGHT - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return scene


def _exif(software: str, original: str, modified: str) -> Image.Exif:
    exif = Image.Exif()
    exif[0x010F] = "ImageSleuth-Synthetic"  # Make
    exif[0x0110] = "DemoGenerator v1"  # Model
    exif[0x0131] = software  # Software
    exif[0x0132] = modified  # DateTime
    exif[0x010E] = LABEL  # ImageDescription
    # Sub-IFDs must be assigned as a dict to be serialised by Pillow.
    exif[0x8769] = {0x9003: original, 0x9004: original}  # Exif IFD: DateTimeOriginal, DateTimeDigitized
    return exif


def _jpeg_bytes(rgb: np.ndarray, quality: int, exif: Image.Exif | None = None) -> bytes:
    buffer = io.BytesIO()
    kwargs = {"exif": exif.tobytes()} if exif is not None else {}
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality, subsampling=2, **kwargs)
    return buffer.getvalue()


def _decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.array(image.convert("RGB"))


def generate(out_dir: Path, seed: int = 2026) -> dict:
    """Write all demo files to ``out_dir`` and return the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = make_scene(seed)
    stamp = "2026:01:15 10:30:00"
    manifest: dict = {"label": LABEL, "seed": seed, "files": {}}

    # 1. Unedited synthetic "original".
    original_jpeg = _jpeg_bytes(scene, 92, _exif("ImageSleuth demo generator (synthetic)", stamp, stamp))
    (out_dir / "01_synthetic_original.jpg").write_bytes(original_jpeg)
    manifest["files"]["01_synthetic_original.jpg"] = {
        "edit": "none", "notes": "Baseline synthetic image saved once as JPEG quality 92 with synthetic EXIF."}

    # 2. Copy-move: duplicate a textured block inside the same image.
    base = _decode(original_jpeg)
    sx, sy, w, h = 110, 470, 200, 160
    tx, ty = 690, 90
    copy_move = base.copy()
    copy_move[ty:ty + h, tx:tx + w] = base[sy:sy + h, sx:sx + w]
    (out_dir / "02_synthetic_copy_move.jpg").write_bytes(
        _jpeg_bytes(copy_move, 92, _exif("ImageSleuth demo generator (synthetic)", stamp, stamp)))
    manifest["files"]["02_synthetic_copy_move.jpg"] = {
        "edit": "copy-move",
        "source_box_xywh": [sx, sy, w, h],
        "target_box_xywh": [tx, ty, w, h],
        "notes": "Block copied within the image, then re-saved at quality 92. EXIF deliberately left looking "
                 "untouched to show that metadata alone does not reveal edits.",
    }

    # 3. Splice: paste never-compressed foreign content into a background previously saved at quality 75
    #    (e.g. an image downloaded from the web), then save the composite at quality 90.
    background = _decode(_jpeg_bytes(scene, 75))
    foreign = make_scene(seed + 99)
    px, py, pw, ph = 560, 400, 224, 192
    spliced = background.copy()
    spliced[py:py + ph, px:px + pw] = foreign[120:120 + ph, 300:300 + pw]
    (out_dir / "03_synthetic_spliced.jpg").write_bytes(
        _jpeg_bytes(spliced, 90, _exif("GIMP 2.10 (simulated tag - synthetic demo)", stamp, "2026:02:03 18:45:12")))
    manifest["files"]["03_synthetic_spliced.jpg"] = {
        "edit": "splice",
        "pasted_box_xywh": [px, py, pw, ph],
        "background_previous_quality": 75,
        "final_quality": 90,
        "notes": "Background previously compressed at quality 75; a block of never-compressed content from a "
                 "different synthetic scene was pasted in and the composite saved at quality 90. Run ELA at "
                 "quality 75 (the background's earlier quality) to see the pasted block stand out; at the default "
                 "90 it is barely visible - a deliberate teaching point. The Software and DateTime EXIF tags "
                 "were set to SIMULATE an editor so the metadata indicators can be demonstrated.",
    }

    # 4. PNG without metadata (lossless): shows metadata absence and the ELA limitation for PNG.
    Image.fromarray(scene).save(out_dir / "04_synthetic_no_metadata.png", format="PNG")
    manifest["files"]["04_synthetic_no_metadata.png"] = {
        "edit": "none", "notes": "Lossless PNG with no EXIF; ELA is only a labelled experiment for PNG."}

    # 5. Naturally repetitive pattern: an honest example of a copy-move FALSE POSITIVE risk.
    rng = np.random.default_rng(seed + 5)
    tile = np.clip(rng.normal(128, 50, (64, 64, 3)), 0, 255).astype(np.uint8)
    tile = cv2.GaussianBlur(tile, (3, 3), 0)
    pattern = np.tile(tile, (HEIGHT // 64, WIDTH // 64, 1))
    (out_dir / "05_repetitive_pattern_unedited.jpg").write_bytes(_jpeg_bytes(pattern, 92))
    manifest["files"]["05_repetitive_pattern_unedited.jpg"] = {
        "edit": "none",
        "notes": "UNEDITED tiled pattern. Repeating content is matched by copy-move analysis even though nothing "
                 "was copied after creation - a deliberate demonstration of a false-positive scenario.",
    }

    # 6. Malformed files for the error-handling demo.
    (out_dir / "06_corrupt_truncated.jpg").write_bytes(original_jpeg[: len(original_jpeg) // 3])
    (out_dir / "07_not_an_image.png").write_bytes(b"This is a plain text file renamed to .png for the demo.\n")
    manifest["files"]["06_corrupt_truncated.jpg"] = {"edit": "corrupt", "notes": "First third of a JPEG only."}
    manifest["files"]["07_not_an_image.png"] = {"edit": "corrupt", "notes": "Text content with a .png extension."}

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "README.txt").write_text(
        LABEL + "\n\nAll files in this folder were generated by scripts/generate_demo_images.py.\n"
        "They are teaching aids only and must not be used to claim real-world detection accuracy.\n"
        "See manifest.json for the exact edits and their pixel locations.\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "assets" / "demo")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    manifest = generate(args.out, args.seed)
    print(f"Wrote {len(manifest['files'])} synthetic demo files to {args.out.resolve()}")
    for name, entry in manifest["files"].items():
        print(f"  {name:38s} {entry['edit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
