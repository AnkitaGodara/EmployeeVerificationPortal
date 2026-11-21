import time

import numpy as np
from PIL import Image

from tests.conftest import jpg, real_face


def test_face_detected_and_cropped():
    from app.face.detector import extract_face
    r = extract_face(jpg(real_face()))
    assert r.found and r.face.shape == (100, 100, 3)


def test_blank_image_has_no_face():
    from app.face.detector import extract_face
    r = extract_face(jpg(Image.new("RGB", (400, 300), (128, 128, 128))))
    assert r.face is None


def test_model_rescales_inside(tmp_path):
    """Normalisation lives inside the model, so raw 0-255 pixels are the correct input."""
    from app.face.model import build_model
    m = build_model(3)
    assert m.layers[0].__class__.__name__ == "Rescaling"


def test_verify_before_training_is_refused(client, admin_headers):
    r = client.post("/api/verify", headers=admin_headers, files={"file": ("x.jpg", jpg(real_face()), "image/jpeg")})
    assert r.status_code == 409


def test_enrol_train_verify(client, admin_headers, people_zip):
    r = client.post("/api/enroll/zip", headers=admin_headers, files={"file": ("dataset.zip", people_zip, "application/zip")})
    assert r.status_code == 200, r.text
    people = {p["label"]: p for p in r.json()["people"]}
    assert set(people) == {"person_a", "person_b", "person_c"}
    assert all(p["saved"] == 12 and p["created"] for p in people.values())

    emp_a = people["person_a"]["employee_id"]
    r = client.patch(f"/api/employees/{emp_a}", headers=admin_headers,
                     json={"name": "Asha Rao", "access_level": "L3", "department": "R&D", "status": "Active"})
    assert r.status_code == 200

    assert client.post("/api/train?epochs=12", headers=admin_headers).status_code == 202
    for _ in range(240):
        s = client.get("/api/train/status", headers=admin_headers).json()
        if s["state"] != "running":
            break
        time.sleep(0.5)
    assert s["state"] == "done", s
    metrics = client.get("/api/train/metrics", headers=admin_headers).json()
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "confusion_matrix"):
        assert key in metrics

    r = client.post("/api/verify", headers=admin_headers, files={"file": ("p.jpg", jpg(real_face()), "image/jpeg")})
    v = r.json()
    assert v["result"] == "MATCH", v
    assert v["employee"]["name"] == "Asha Rao"
    assert "R&D lab" in v["allowed_zones"] and "Server room (data centre)" not in v["allowed_zones"]

    r = client.post("/api/verify", headers=admin_headers,
                    files={"file": ("b.jpg", jpg(Image.new("RGB", (640, 480), (90, 90, 90))), "image/jpeg")})
    assert r.json()["result"] == "NO_FACE"

    logs = client.get("/api/logs", headers=admin_headers).json()
    assert {l["result"] for l in logs} >= {"MATCH", "NO_FACE"}


def test_suspended_employee_gets_warning_and_no_zones(client, admin_headers):
    emp = next(e for e in client.get("/api/employees", headers=admin_headers).json() if e["label"] == "person_a")
    client.patch(f"/api/employees/{emp['id']}", headers=admin_headers, json={"status": "Suspended"})
    v = client.post("/api/verify", headers=admin_headers, files={"file": ("p.jpg", jpg(real_face()), "image/jpeg")}).json()
    assert v["warning"] and "SUSPENDED" in v["warning"]
    assert v["allowed_zones"] == []
    client.patch(f"/api/employees/{emp['id']}", headers=admin_headers, json={"status": "Active"})


def test_zip_without_person_folders_is_rejected(client, admin_headers):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("photo.jpg", jpg(real_face()))
    r = client.post("/api/enroll/zip", headers=admin_headers, files={"file": ("x.zip", buf.getvalue(), "application/zip")})
    assert r.status_code == 400


def test_zip_path_traversal_names_stay_inside_data_dir(client, admin_headers):
    import io
    import zipfile
    from app.config import settings
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        arr = np.full((100, 100, 3), 120, "uint8")
        zf.writestr("../../evil/x.jpg", jpg(Image.fromarray(arr)))
    client.post("/api/enroll/zip", headers=admin_headers, files={"file": ("x.zip", buf.getvalue(), "application/zip")})
    assert (settings.FACES_DIR / "evil").exists()           # sanitised into a normal label folder
    assert not (settings.DATA_DIR.parent / "evil").exists()
