"""Writes notebooks/f5_colab.ipynb (F5 fine-tuning on a free Colab T4, resumable via MinIO; K60).

Run:  .venv/bin/python scripts/f5_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    ("md", """# Jev F5 — fine-tuning B2 / B3 / C1 di Colab T4 (bisa dilanjutkan)

**Unggah ke `/content`** (panel *Files* di kiri):
1. `jev_f5_bundle.zip` — kode `jevexp`, `splits/` (termasuk `f5_plan.json`), `data/MANIFEST.json`
2. `config.json` — kredensial MinIO (tidak pernah dicetak; hanya sidik jarinya)

**Secret Colab:** `HF_TOKEN` (opsional; semua model publik).
**Runtime:** *Change runtime type* → **T4 GPU**. Lalu **Runtime → Run all**.

**Kalau Colab mati / terputus:** sambungkan ulang, unggah lagi dua file di atas bila hilang, lalu **Run all**.
MinIO adalah satu-satunya sumber kebenaran:
- run yang sudah punya `done.json` dilewati;
- run yang terputus dilanjutkan dari checkpoint epoch terakhirnya (disimpan tiap ±15 menit);
- data diunduh ulang (±1 menit), model dasar diunduh ulang dari Hugging Face.

Notebook ini **tidak pernah menghitung metrik test** (aturan pra-registrasi: tanpa *interim look*). Yang dicetak
hanya macro-F1 **validasi**. Prediksi test disimpan untuk analisis F6 di laptop.
"""),
    ("md", "## Cell 1 — Pengaturan"),
    ("code", """# ===== Cell 1: Pengaturan =====
from pathlib import Path

STAGE = 1              # stage AWAL: 1 = inti (grid, B2/B3 penuh, C1 + C2) · 2 = kurva NusaX · 3 = kurva NT
AUTO_NEXT = True       # selesai satu stage -> langsung lanjut stage berikutnya (sel tetap sibuk = keep alive)
STATUS_EVERY_MIN = 10  # blok status + estimasi dicetak tiap sekian menit (dan setiap kali satu run selesai)
BUCKET = "private"
S3_PREFIX = "jev_f5"
CKPT_EVERY_MIN = 15    # checkpoint tengah-run paling sering tiap sekian menit (di akhir epoch)
SHARED = False         # True hanya bila DUA sesi Colab menjalankan tahap yang sama bersamaan
ALLOW_NEW_CODE = True  # bundle ini membawa perbaikan K62 + status K63 (tercatat di log keputusan)

CONTENT = Path("/content")
JEV = CONTENT / "jev"
WORK = CONTENT / "work"
OUTPUT = CONTENT / "output"
CONFIG_CANDIDATES = ("config.json", "/content/config.json")
for p in (WORK, OUTPUT):
    p.mkdir(parents=True, exist_ok=True)
"""),
    ("md", "## Cell 2 — Instal paket, ekstrak bundle, cek GPU"),
    ("code", """# ===== Cell 2: Instal & bundle =====
import os, shutil, subprocess, sys, zipfile

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

pip("laya==0.3.21", "transformers==5.17.0", "boto3")

bundle = CONTENT / "jev_f5_bundle.zip"
assert bundle.exists(), "Unggah jev_f5_bundle.zip ke /content dulu"
if JEV.exists():
    shutil.rmtree(JEV)                      # selalu segar dari bundle yang diunggah
zipfile.ZipFile(bundle).extractall(CONTENT)
sys.path.insert(0, str(JEV / "src"))
print((JEV / "PREREG_TAG").read_text().strip())

import torch
assert torch.cuda.is_available(), "GPU tidak aktif: Runtime → Change runtime type → T4 GPU"
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
"""),
    ("md", "## Cell 3 — Token Hugging Face & MinIO"),
    ("code", """# ===== Cell 3: HF_TOKEN & MinIO =====
def get_hf_token():
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN") or None
    except Exception:
        return os.environ.get("HF_TOKEN") or None

HF_TOKEN = get_hf_token()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
print("HF_TOKEN:", "ada" if HF_TOKEN else "tidak ada (model publik tetap bisa diunduh)")

from jevexp.ft.store import s3_store
config_path = next((p for p in CONFIG_CANDIDATES if Path(p).is_file()), None)
assert config_path, "config.json tidak ditemukan; unggah ke /content"
STORE, where = s3_store(config_path, bucket=BUCKET, prefix=S3_PREFIX)
print(f"MinIO terhubung: {where}, bucket '{BUCKET}', prefix '{S3_PREFIX}'")
"""),
    ("md", "## Cell 4 — Data (diunduh dari GitHub pada commit terkunci, diverifikasi sha256)"),
    ("code", """# ===== Cell 4: Data =====
from jevexp import data as jdata
manifest = jdata.fetch()          # gagal keras bila satu hash tidak sama dengan MANIFEST
print(len(manifest), "file data terverifikasi")
"""),
    ("md", """## Cell 5 — Jalankan + pantau status (cell terakhir, sengaja blocking = keep alive)

Sel ini mengerjakan stage `STAGE`, lalu (bila `AUTO_NEXT`) langsung stage berikutnya sampai stage 3 habis.
Selama berjalan, blok status dicetak setiap kali satu run selesai dan tiap `STATUS_EVERY_MIN` menit:

```
==== STATUS F5 · 2026-09-28 07:10 UTC · sesi 3f2a… (1.4 j, 6 run selesai) ====
Stage 1 [##..................]   25/250  selesai · sisa ±21.3 j  <- sekarang
    grid B2 nusax_senti grid               20/20
    train C1 nt_emot pooled                 1/3
    ...
Stage 2 [....................]    0/432  selesai · sisa ±12.6 j*
Stage 3 [....................]    0/588  selesai · sisa ±17.1 j*
Sedang: c1-nt_emot-pooled-s1-nfull — epoch 2/4 loss 0.41 (berjalan 18 mnt)
Total sisa semua stage: ±51.0 j  (* = sebagian masih perkiraan awal …)
```

`*` = estimasi masih memakai perkiraan awal; angka makin akurat setelah run pertama di kelompok itu selesai.
Blok yang sama disimpan ke `/content/output/progress.txt` dan ke MinIO `jev_f5/status/progress.txt`
(dari laptop: `python -m jevexp.ft.progress --config <config.json>`).

Setiap stage selesai, hasil otomatis dikumpulkan ke `/content/output` (`f5_results.zip` + `runs/`, `meta/`,
`status.csv`). Mengumpulkan manual dari sesi baru: jalankan Cell 1–3, lalu
`Worker(STORE, work=WORK).collect(OUTPUT)`.
"""),
    ("code", """# ===== Cell 5: Jalankan semua stage + status =====
from jevexp.ft.worker import Worker

W = Worker(STORE, work=WORK, stage=STAGE, hf_token=HF_TOKEN, ckpt_every_s=CKPT_EVERY_MIN * 60,
           shared=SHARED, status_every_s=STATUS_EVERY_MIN * 60, out_dir=OUTPUT)
print(W.check_plan(allow_new_code=ALLOW_NEW_CODE))
W.run_all(STAGE, auto_next=AUTO_NEXT)
"""),
]


def main():
    cells = []
    for kind, src in CELLS:
        lines = src.strip("\n").splitlines(keepends=True)
        cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": lines}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = ROOT / "notebooks" / "f5_colab.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(out)


if __name__ == "__main__":
    main()
