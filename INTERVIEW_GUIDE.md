# Interview guide: Employee Verification Portal

## 1. The 60-second pitch

"I built a full-stack employee verification portal. A security officer logs in with a user ID and
password, uploads or captures a photo, and my CNN recognises the employee and shows their details,
access level and the zones they're allowed into. If the model isn't confident, it says UNKNOWN
instead of guessing. On top of that there's a RAG assistant: the officer can ask 'can this person
enter the server room?' and it answers from the company's policy documents and the employee's
record, with citations. Admins can upload a folder of photos per person, retrain the model from the
UI and see accuracy, precision, recall, F1 and a confusion matrix. It's React plus FastAPI,
TensorFlow, OpenCV, LangChain and FAISS, with 30 automated tests, and it's deployed live on Hugging Face Spaces."

## 2. Walk through the workflow (draw this)

1. **Login:** the password is checked against a bcrypt hash and the server returns a JWT. Every later request sends it; roles decide what you can do (admin vs security).
2. **Upload:** the photo goes to `/api/verify`.
3. **Detect:** OpenCV Haar cascade finds the largest face, crops a square with margin, resizes to 100×100. *The same function prepares training images*, so training and prediction inputs match.
4. **Predict:** the CNN outputs softmax probabilities, and I take the top 3.
5. **Decide:** top probability ≥ 70% → MATCH, otherwise UNKNOWN. No face → NO_FACE.
6. **Card:** look up the employee by the predicted label and compute allowed zones from their access level and status.
7. **Log:** every attempt is written to the audit log.
8. **Ask:** the RAG assistant gets the question plus the employee id.

## 3. Design decisions: what and why

**The CNN**
- Four conv blocks (32-64-128-256 filters, 4×4 kernels, max-pooling), then Dense 128 → Dropout 0.3 → Dense 64 → softmax.
- **I moved normalisation (Rescaling 1/255) inside the model.** In my first notebook I normalised only at prediction time, not in training, so the model saw different scales. Putting it in the model makes that bug impossible.
- **Augmentation layers inside the model** (flip, rotation, zoom, shift): with ~10 photos per person this is the main defence against overfitting. Keras turns them off automatically at prediction time.
- Stratified 80/20 split, balanced class weights, early stopping on validation loss with best weights restored.

**Why UNKNOWN matters (open-set recognition)**
A softmax classifier *always* picks one of the known people, even for a stranger. So I added a confidence threshold: below 70% the portal says UNKNOWN and the policy tells the guard to check the ID card manually. That's the difference between a demo and something you could use at a gate.

**Why Haar cascade for detection?** It ships with OpenCV (no model download), runs on CPU in milliseconds and works for frontal faces at a desk. The trade-off: it misses side profiles. The next step would be a DNN detector such as YuNet or MTCNN.

**The RAG pipeline**
1. **Knowledge base:** 5 policy documents and one profile document per employee (generated from the database, so it's always current).
2. **Chunking:** split by markdown headings first, so each chunk stays inside one section, then into ~700-character pieces with 100 overlap. Every chunk is prefixed with "[Document > Section]" so it makes sense on its own.
3. **Hybrid retrieval:** FAISS vector search plus BM25 keyword search, fused with **Reciprocal Rank Fusion** (score = Σ 1/(60 + rank)). Vectors catch meaning; BM25 catches exact terms like "L4" or "Unknown".
4. **Person-aware context:** when a verified employee is selected, their record is fetched *by id* (not left to similarity search) and always included as source S1.
5. **Deterministic rule check:** for "can X enter <zone>" questions I compute the answer from the zone table (zone needs L4, employee is L3, so NOT ALLOWED) and give it to the LLM as a cited source. **The yes/no decision never depends on the LLM's reasoning.**
6. **Grounded generation:** the system prompt says to answer only from the numbered sources, cite [S1]…, and say "I couldn't find this in the company documents" otherwise. Temperature 0.1.
7. **Fallbacks:** no API key or the LLM call fails → extractive answer (the most relevant policy sentences, with citations), so the app never breaks.

**I measured retrieval instead of guessing.** 28 labelled questions, each mapped to the policy section that answers it:

| Retriever | Hit@4 | MRR |
|---|---|---|
| Vector only | 75% | 0.64 |
| BM25 only | 79% | 0.66 |
| Hybrid | **82%** | **0.67** |

The remaining misses are vocabulary gaps ("refuse my photo" vs "withdraw consent"). That is what semantic embeddings fix; the code switches with one setting and the same script measures it.

**Security**
- bcrypt password hashes, JWT with expiry, role checks on every admin route.
- Lockout after 5 failed logins (5 minutes).
- Zip uploads are read in memory and labels are sanitised, so a file named `../../x.jpg` can't escape the data folder (zip-slip), and there's a test for it.
- Uploaded markdown is escaped before rendering (no script injection).
- Photos and the database are git-ignored; deleting an employee deletes their photos.

## 4. Questions you'll likely get

**Q: Why train a CNN from scratch instead of using FaceNet / ArcFace embeddings?**
A: This started as my CNN project, so I kept and fixed that model. For production I'd switch to a pretrained face-embedding model: you store one embedding per person and compare with cosine similarity, so adding a new employee needs no retraining and it works with 1–5 photos. My architecture already separates detection, recognition and the decision rule, so only `model.py` would change.

**Q: Your accuracy is 100% / very high. Do you trust it?**
A: Not fully. With ~10 photos per person the validation set has only ~2 images per person, so one mistake moves accuracy by several points. I show the confusion matrix and per-person scores for that reason, and the UI warns when the validation set is small. Cross-validation or a separate test set of new photos would give a more reliable number.

**Q: How do you stop the LLM from hallucinating?**
A: Four layers: (1) it only gets retrieved sources and is told to answer only from them; (2) every claim needs a [S#] citation, and the UI shows the exact source text; (3) critical decisions like zone access are computed by code, not by the LLM; (4) if it can't find the answer it must say so.

**Q: Why FAISS and not a vector database like Pinecone?**
A: The knowledge base is ~50 chunks. FAISS in-process is instant, free and has no extra service to run. For thousands of documents or multiple servers I'd move to pgvector or a managed vector DB.

**Q: Why hybrid search?**
A: Policy text is full of exact codes (L3, L4, 70%, extension numbers) that keyword search handles well, while vector search handles paraphrases. RRF combines rankings without having to calibrate two different score scales. It raised Hit@4 from 75% to 82%.

**Q: What happens when someone new joins?**
A: The admin uploads a folder of their photos (faces are detected and cropped automatically), fills in their details and clicks Train. Training runs in a background thread; when it finishes, the new model is swapped in without restarting the server.

**Q: How would you scale this?**
A: Move training to a job queue (Celery or RQ) instead of a thread, PostgreSQL instead of SQLite, store photos in object storage, and switch to face embeddings so enrolment doesn't need retraining. Add a liveness check (blink or depth) so a printed photo can't fool the camera.

**Q: What was the hardest bug?**
A: The normalisation mismatch between training and prediction in my first version: predictions were wrong even though training accuracy looked fine. Moving Rescaling into the model fixed it for good. (A smaller one: OpenCV 5 stopped shipping the Haar cascade files, so I pinned OpenCV 4.x.)

## 5. Limitations (say these before they ask)

- A CNN trained from scratch needs retraining for every new person, and accuracy depends heavily on photo quality.
- Haar cascade struggles with side faces, masks and poor light.
- No liveness detection: a photo of a photo could pass.
- The evaluation set is small (28 questions; ~2 validation images per person).

## 6. Resume lines (fill in your real numbers after training)

- Built a full-stack **employee verification portal** (React, FastAPI, SQLite) with JWT login, role-based access and an audit log; face recognition via a **4-layer Keras CNN** with OpenCV face detection, augmentation and a confidence threshold that rejects unknown faces.
- Trained on **[N] images of [K] people**, reaching **[X]% validation accuracy / [Y] macro-F1**, tracked with a confusion matrix and per-person precision/recall.
- Added a **RAG policy assistant** (LangChain, FAISS + BM25 hybrid retrieval with RRF, Groq LLM) that answers access and HR questions with citations; hybrid retrieval improved **Hit@4 from 75% to 82%** on a 28-question labelled set.
