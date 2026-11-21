import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useAuth } from "../App.jsx";
import { Notice, PageHeader } from "../components/ui.jsx";

const EMPTY = { user_id: "", full_name: "", password: "", role: "security" };

export default function Users() {
  const { user: me } = useAuth();
  const [rows, setRows] = useState([]);
  const [form, setForm] = useState(EMPTY);
  const [notice, setNotice] = useState(null);
  const load = () => api.get("/api/users").then(setRows).catch(() => {});
  useEffect(() => { load(); }, []);
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  async function create(e) {
    e.preventDefault();
    try {
      await api.post("/api/users", form);
      setNotice({ kind: "info", text: `User ${form.user_id} created.` });
      setForm(EMPTY);
      load();
    } catch (err) {
      setNotice({ kind: "error", text: err.message });
    }
  }

  async function toggle(u) {
    try {
      await api.patch(`/api/users/${encodeURIComponent(u.user_id)}/active?active=${!u.is_active}`);
      load();
    } catch (err) {
      setNotice({ kind: "error", text: err.message });
    }
  }

  return (
    <>
      <PageHeader title="Portal users" subtitle="Who can sign in. Security users can verify and ask the assistant; admins can also enrol, train and see logs." />
      {notice && <Notice kind={notice.kind} onClose={() => setNotice(null)}>{notice.text}</Notice>}
      <form className="card form-grid" onSubmit={create}>
        <h2 className="section-title span-2">Add user</h2>
        <label>User ID<input value={form.user_id} onChange={set("user_id")} required minLength={3} /></label>
        <label>Full name<input value={form.full_name} onChange={set("full_name")} required /></label>
        <label>Password (8+ characters)<input type="password" value={form.password} onChange={set("password")} required minLength={8} autoComplete="new-password" /></label>
        <label>Role
          <select value={form.role} onChange={set("role")}>
            <option value="security">Security (verify + assistant)</option>
            <option value="admin">Admin (everything)</option>
          </select>
        </label>
        <div className="span-2 row-end"><button className="btn primary">Create user</button></div>
      </form>
      <div className="card table-card">
        <table>
          <thead><tr><th>User ID</th><th>Name</th><th>Role</th><th>Status</th><th></th></tr></thead>
          <tbody>
            {rows.map((u) => (
              <tr key={u.user_id}>
                <td>{u.user_id}</td><td>{u.full_name}</td><td>{u.role}</td>
                <td>{u.is_active ? <span className="badge good">✓ Active</span> : <span className="badge neutral">– Disabled</span>}</td>
                <td className="row-actions">
                  {u.user_id !== me.user_id && (
                    <button className="btn small ghost" onClick={() => toggle(u)}>{u.is_active ? "Disable" : "Enable"}</button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
