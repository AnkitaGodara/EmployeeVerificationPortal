"""Command-line helpers (run from the backend folder):

  python -m scripts.manage import-zip "path/to/project 1.zip"   # enrol a folder-per-person zip
  python -m scripts.manage import-folder "path/to/project 1"     # same, from an unzipped folder
  python -m scripts.manage details scripts/employees.sample.json # fill employee details by label
  python -m scripts.manage train --epochs 40                     # train the CNN (prints metrics)
  python -m scripts.manage add-user security1 "Gate Security" Security@123 --role security
  python -m scripts.manage reindex                               # rebuild the RAG index
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

from app.db import Employee, SessionLocal, User, init_db
from app.face.dataset import IMAGE_EXTS, group_zip_by_person, ingest_images, save_profile_photo
from app.security import ensure_default_admin, hash_password


def next_code(db):
    n = db.query(Employee).count() + 1
    while db.query(Employee).filter(Employee.employee_code == f"ONX{n:04d}").first():
        n += 1
    return f"ONX{n:04d}"


def enrol_groups(groups: dict):
    db = SessionLocal()
    try:
        for label, images in sorted(groups.items()):
            emp = db.query(Employee).filter(Employee.label == label).first()
            if emp is None:
                emp = Employee(label=label, employee_code=next_code(db), name=label.replace("_", " ").title())
                db.add(emp)
                db.commit()
            report = ingest_images(label, images)
            first = report.pop("first_good_image")
            if first and not emp.photo_path:
                emp.photo_path = save_profile_photo(label, first)
                db.commit()
            print(f"{label:15s} saved {report['saved']:3d}  skipped {report['skipped']:3d}  "
                  f"total faces {report['total_faces']}")
            for f in report["files"]:
                if f["status"] == "skipped":
                    print(f"    skipped {f['file']}: {f['reason']}")
    finally:
        db.close()


def folder_to_groups(root: Path) -> dict:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                zf.write(p, p.relative_to(root.parent).as_posix())
    return group_zip_by_person(buf.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("import-zip").add_argument("path")
    sub.add_parser("import-folder").add_argument("path")
    sub.add_parser("details").add_argument("path")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=None)
    u = sub.add_parser("add-user")
    u.add_argument("user_id")
    u.add_argument("full_name")
    u.add_argument("password")
    u.add_argument("--role", default="security", choices=["admin", "security"])
    sub.add_parser("reindex")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    ensure_default_admin(db)
    db.close()

    if args.cmd == "import-zip":
        enrol_groups(group_zip_by_person(Path(args.path).read_bytes()))
    elif args.cmd == "import-folder":
        enrol_groups(folder_to_groups(Path(args.path)))
    elif args.cmd == "details":
        db = SessionLocal()
        for row in json.loads(Path(args.path).read_text()):
            emp = db.query(Employee).filter(Employee.label == row["label"]).first()
            if emp is None:
                emp = Employee(label=row["label"], employee_code=next_code(db), name=row.get("name", row["label"]))
                db.add(emp)
            for k, v in row.items():
                if k != "label" and hasattr(emp, k):
                    setattr(emp, k, v)
            db.commit()
            print("updated", row["label"])
        db.close()
    elif args.cmd == "train":
        from app.face import model as cnn
        from app.face.dataset import load_training_set
        images, labels = load_training_set()
        print(f"Training on {len(labels)} images of {len(set(labels))} people...")
        m = cnn.train(images, labels, epochs=args.epochs,
                      progress=lambda e, n, logs: print(f"  epoch {e}/{n}  loss {logs.get('loss', 0):.3f}  "
                                                        f"val_acc {logs.get('val_accuracy', 0):.2%}"))
        print(json.dumps({k: m[k] for k in ("classes", "accuracy", "precision_macro", "recall_macro",
                                            "f1_macro", "train_images", "val_images", "dropped_classes")}, indent=2))
    elif args.cmd == "add-user":
        db = SessionLocal()
        if db.query(User).filter(User.user_id == args.user_id).first():
            print("user already exists")
        else:
            db.add(User(user_id=args.user_id, full_name=args.full_name,
                        password_hash=hash_password(args.password), role=args.role))
            db.commit()
            print("created", args.user_id, args.role)
        db.close()
    if args.cmd in ("import-zip", "import-folder", "details", "reindex"):
        from app.rag.index import rag_index
        print("RAG index:", rag_index.build())


if __name__ == "__main__":
    main()
