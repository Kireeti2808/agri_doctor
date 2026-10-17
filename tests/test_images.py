"""Upload validation tests (no network, no models)."""

import io

import pytest
from PIL import Image

import agri_core as core


def encode(image, fmt="JPEG", exif=None):
    buffer = io.BytesIO()
    kwargs = {"exif": exif} if exif is not None else {}
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def test_exif_rotation_is_applied():
    # Stored landscape 200x100 with "rotate 90" orientation -> shown portrait.
    stored = Image.new("RGB", (200, 100), (10, 120, 30))
    exif = Image.Exif()
    exif[0x0112] = 6
    image, notes = core.load_leaf_image(encode(stored, exif=exif.tobytes()))
    assert image.size == (100, 200)
    assert image.mode == "RGB"


def test_png_with_alpha_becomes_rgb():
    image, _ = core.load_leaf_image(encode(Image.new("RGBA", (128, 128)), "PNG"))
    assert image.mode == "RGB"


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20])
def test_unreadable_files_are_rejected(data):
    with pytest.raises(core.ImageInputError, match="could not be read"):
        core.load_leaf_image(data)


def test_tiny_images_are_rejected():
    with pytest.raises(core.ImageInputError, match="at least"):
        core.load_leaf_image(encode(Image.new("RGB", (40, 300))))


def test_huge_images_are_scaled_down():
    image, notes = core.load_leaf_image(encode(Image.new("RGB", (6000, 3000))))
    assert max(image.size) == core.MAX_IMAGE_SIDE
    assert notes
