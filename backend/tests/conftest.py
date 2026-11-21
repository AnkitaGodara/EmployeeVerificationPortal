import io
import os
import tempfile
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageEnhance

# Use a throw-away data folder and no LLM for every test run (set before the app is imported).
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="evp_test_")
os.environ["LLM_PROVIDER"] = "none"
os.environ["EPOCHS"] = "12"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def jpg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def real_face() -> Image.Image:
    from skimage import data   # public-domain sample photo with one clear face
    return Image.fromarray(data.astronaut())


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def admin_headers(client):
    r = client.post("/api/auth/login", data={"username": "admin", "password": "Admin@123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="session")
def people_zip() -> bytes:
    """person_a: 12 variants of a real face photo. person_b / person_c: small pre-cropped images."""
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        base = real_face()
        for i in range(12):
            im = ImageEnhance.Brightness(base.rotate(rng.uniform(-8, 8))).enhance(rng.uniform(0.75, 1.25))
            if i % 2:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            zf.writestr(f"dataset/person_a/a{i}.jpg", jpg(im))
        for ci, name in enumerate(["person_b", "person_c"]):
            for i in range(12):
                arr = np.zeros((100, 100, 3))
                arr[..., ci + 1] = 170
                arr = np.clip(arr + rng.normal(0, 20, arr.shape), 0, 255).astype("uint8")
                zf.writestr(f"dataset/{name}/{name}_{i}.png", jpg(Image.fromarray(arr)))
        zf.writestr("dataset/readme.txt", "not an image")
        zf.writestr("loose_root_image.jpg", jpg(base))      # no person folder -> ignored
    return buf.getvalue()
