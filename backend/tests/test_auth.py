def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_wrong_password_rejected(client):
    r = client.post("/api/auth/login", data={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_protected_routes_need_login(client):
    assert client.get("/api/employees").status_code == 401
    assert client.post("/api/verify").status_code == 401


def test_me(client, admin_headers):
    me = client.get("/api/auth/me", headers=admin_headers).json()
    assert me["user_id"] == "admin" and me["role"] == "admin"


def test_security_user_cannot_use_admin_routes(client, admin_headers):
    r = client.post("/api/users", headers=admin_headers,
                    json={"user_id": "guard1", "full_name": "Gate Guard", "password": "Guard@1234", "role": "security"})
    assert r.status_code == 201
    tok = client.post("/api/auth/login", data={"username": "guard1", "password": "Guard@1234"}).json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    assert client.get("/api/employees", headers=h).status_code == 403
    assert client.post("/api/train", headers=h).status_code == 403
    assert client.get("/api/logs", headers=h).status_code == 403
    assert client.get("/api/documents", headers=h).status_code == 200


def test_lockout_after_five_failures(client, admin_headers):
    client.post("/api/users", headers=admin_headers,
                json={"user_id": "locktest", "full_name": "Lock Test", "password": "Right@1234"})
    for _ in range(5):
        client.post("/api/auth/login", data={"username": "locktest", "password": "bad"})
    r = client.post("/api/auth/login", data={"username": "locktest", "password": "Right@1234"})
    assert r.status_code == 429


def test_disabled_user_cannot_login(client, admin_headers):
    client.post("/api/users", headers=admin_headers,
                json={"user_id": "temp1", "full_name": "Temp", "password": "Temp@12345"})
    client.patch("/api/users/temp1/active?active=false", headers=admin_headers)
    r = client.post("/api/auth/login", data={"username": "temp1", "password": "Temp@12345"})
    assert r.status_code == 403
