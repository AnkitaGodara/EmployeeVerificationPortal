import json


def test_seed_from_zip_and_demo_user(client, people_zip, tmp_path, monkeypatch):
    from app.db import Employee, SessionLocal, User
    from app.seed import ensure_demo_user, seed_from_zip

    zip_path = tmp_path / "seed.zip"
    zip_path.write_bytes(people_zip)
    details = tmp_path / "details.json"
    details.write_text(json.dumps([{"label": "person_b", "name": "Bala", "access_level": "L4", "department": "IT"}]))

    summary = seed_from_zip(zip_path.as_uri(), details_path=details)
    assert summary["person_b"] == 12
    db = SessionLocal()
    emp = db.query(Employee).filter(Employee.label == "person_b").first()
    assert emp.name == "Bala" and emp.access_level == "L4"

    monkeypatch.setenv("DEMO_USER_ID", "demo")
    monkeypatch.setenv("DEMO_PASSWORD", "Demo@12345")
    ensure_demo_user()
    assert db.query(User).filter(User.user_id == "demo", User.role == "security").first()
    db.close()
    r = client.post("/api/auth/login", data={"username": "demo", "password": "Demo@12345"})
    assert r.status_code == 200 and r.json()["role"] == "security"
