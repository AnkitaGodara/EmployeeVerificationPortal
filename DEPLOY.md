# Live demo on Hugging Face Spaces (free)

Why Hugging Face: the free CPU Space has 2 vCPU and 16 GB RAM, enough for TensorFlow. (Free tiers
with 512 MB RAM, like Render's, are too small.) Your link will look like:

```
https://<your-hf-username>-employee-verification-portal.hf.space
```

Total time: about 20 minutes, most of it waiting for the build.

---

## Step 1: Create a Hugging Face account

Sign up at https://huggingface.co/join and verify your email.

## Step 2: Put the photos in a PRIVATE dataset

The photos must never be public, so they go in a private dataset that only your Space can read.

1. Rename your zip to **`faces.zip`** (no spaces). Inside it: one folder per person (`harshal/`, `kishan/` ...).
2. Go to https://huggingface.co/new-dataset, name it **`face-portal-photos`**, choose **Private**, then **Create dataset**.
3. **Files and versions → Add file → Upload files**, drop `faces.zip`, then **Commit**.
4. Your seed URL is:
   ```
   https://huggingface.co/datasets/<your-hf-username>/face-portal-photos/resolve/main/faces.zip
   ```

## Step 3: Create two access tokens

https://huggingface.co/settings/tokens, then **Create new token**:

- **Read** token, name it `space-seed`. The Space uses it to download `faces.zip`.
- **Write** token, name it `git-push`. You use it as the password when pushing code in Step 6.

Copy each token when it's shown; you can't see it again.

## Step 4: Create the Space

https://huggingface.co/new-space:

- Space name: **`employee-verification-portal`**
- SDK: **Docker**, template **Blank**
- Hardware: **CPU basic (free)**
- Visibility: **Public** (needed for a shareable demo link)

## Step 5: Add the settings (Space → Settings → Variables and secrets)

**Secrets** (hidden):

| Name | Value |
|---|---|
| `JWT_SECRET` | any long random text (e.g. 40+ random characters) |
| `ADMIN_PASSWORD` | your admin password (keep this to yourself) |
| `DEMO_PASSWORD` | password for the demo login you share, e.g. `Demo@2026` |
| `GROQ_API_KEY` | from https://console.groq.com/keys |
| `SEED_ZIP_TOKEN` | the **Read** token from Step 3 |

**Variables** (visible):

| Name | Value |
|---|---|
| `DEMO_USER_ID` | `demo` |
| `SEED_ZIP_URL` | the URL from Step 2 |
| `LLM_PROVIDER` | `groq` |
| `EPOCHS` | `40` |

## Step 6: Upload the code

In Command Prompt, from the folder containing the unzipped project (`face-portal`):

```bat
git clone https://huggingface.co/spaces/<your-hf-username>/employee-verification-portal hf-space
xcopy /E /I /Y face-portal hf-space
copy /Y face-portal\deploy\huggingface\README.md hf-space\README.md
cd hf-space
git add .
git commit -m "Deploy employee verification portal"
git push
```

When git asks for a password, paste the **Write** token. (Username: your HF username.)

The Space's `README.md` must be the one from `deploy/huggingface/`: its header tells Hugging Face
to build the Dockerfile. Your GitHub README stays unchanged.

## Step 7: Wait for the build, then test

1. Open the Space and watch the **Logs** tab. The build takes 5–10 minutes.
2. When it says **Running**, the app downloads `faces.zip`, enrols everyone and trains (1–3 minutes).
   The log shows `Seeded faces: {...}`.
3. Open the link, sign in as **admin** and check **Enrol & Train** for the metrics.
4. Sign out and sign in as **demo**. This is what interviewers will see.

## Step 8: Add the link everywhere

- GitHub README: fill in the **Live demo** line at the top.
- Resume: add `| Live Demo` next to the Code link on the project.

---

## Good to know

- **Sleeping:** a free Space sleeps after 48 hours without visitors. The first visit wakes it
  (about 1 minute), then it retrains automatically (1–3 minutes). **Open the link 10 minutes before an
  interview.**
- **Changes made in the app are temporary:** employees you add or edit in the UI are lost when the
  Space restarts. Permanent changes go in `faces.zip` (photos) or `backend/scripts/employees.sample.json`
  (details).
- **What interviewers can test:** with the demo login they can use the webcam. They will get
  **UNKNOWN**, which shows the open-set rule working. To show a **MATCH**, verify your own face live
  (if you're in the dataset).
- **Privacy:** anyone with the demo login sees the enrolled people's names and profile photos. Only
  include people who agreed, or use only your own photos.
- **Updating:** push again to the Space (Step 6 commands from `git add .`). It rebuilds and reseeds.
