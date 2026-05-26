# mapping-fer-with-au

Proyek ini memetakan ekspresi wajah dari video DAiSEE ke 4 affective state: **Boredom**, **Confusion**, **Frustration**, dan **Engagement**. Berbeda dari v1 yang pakai emosi dasar (FER), di sini kita pakai **Action Units** (AU) yang lebih granular dan langsung merepresentasikan gerakan otot wajah.

Dataset: [DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html) — 8.566 video clip siswa belajar online, masing-masing ~10 detik.

---

## Kenapa pakai Action Units?

FER (Facial Expression Recognition) mengelompokkan ekspresi ke 7 emosi dasar (happy, sad, angry, dll). Tapi affective state seperti "confusion" tidak selalu terlihat seperti emosi yang jelas — lebih sering muncul dari gerakan otot kecil: alis turun (AU04), mata menyipit (AU07), bibir tegang (AU23).

Action Units dari FACS (Facial Action Coding System) mendeskripsikan gerakan otot wajah secara individual. Dengan 20 AU + statistik temporal per video, model bisa menangkap pola yang lebih spesifik dibanding sekadar rata-rata emosi.

---

## Pipeline

```
video DAiSEE  →  [01] ekstrak AU  →  [02] train XGBoost  →  [03] inference
```

### Script 01 — `01_extract_au_features.py`

Mengambil AU dari setiap frame video menggunakan **py-feat** (RetinaFace untuk deteksi muka, MobileFaceNet untuk landmark, XGBoost untuk AU).

Yang dilakukan script ini:
1. Buka video dengan OpenCV, ambil 1 frame per detik (DAiSEE ~30fps, jadi skip tiap 30 frame)
2. Simpan frame ke folder temp sebagai PNG
3. Kirim batch frame ke `det.detect_image()` — py-feat return DataFrame dengan 20 kolom AU
4. Kalau ada >1 muka per frame (mis. ada orang lewat di background), ambil yang `FaceScore` tertinggi
5. Isi frame tanpa muka dengan forward/backward fill, lalu rolling mean 3 frame untuk smoothing
6. Simpan ke CSV dengan format `video_id, timestamp, AU01, AU02, ..., AU43`

Fitur resume: script cek video mana yang sudah ada di CSV output, skip yang sudah selesai. Aman kalau proses mati di tengah jalan.

```bash
# coba dulu 5 video
python scripts/01_extract_au_features.py --sample 5

# full run (estimasi ~18-30 jam di laptop dengan GPU)
python scripts/01_extract_au_features.py --device cuda

# di Colab A100, pakai batch size lebih besar
python scripts/01_extract_au_features.py --device cuda --batch-size 16
```

Output: `data/au_features/{split}_au_features.csv` dan `{split}_au_features_agg.csv`

---

### Script 02 — `02_calibrate_v2.py`

Training XGBoost classifier untuk masing-masing 4 affective state.

Alurnya:
1. **Feature engineering** — dari frame-level AU per video, hitung 6 statistik temporal: `mean, std, max, min, skew, slope`. Total 20 AU × 6 = 120 fitur + `n_frames` = **121 fitur per video**
2. **Training** — `XGBClassifier` dengan `scale_pos_weight` untuk menangani class imbalance (frustration hanya ~5% dari data)
3. **Threshold tuning** — default threshold 0.5 tidak optimal untuk data imbalanced. Script sweep threshold 0.05–0.95 di validation set dan pilih yang maksimalkan F1
4. **Simpan model** — 4 model XGBoost ke `models/xgb_models.joblib`, threshold ke `models/calibration_v2.json`

```bash
python scripts/02_calibrate_v2.py
```

---

### Script 03 — `03_inference_v2.py`

Load model yang sudah ditraining, jalankan prediksi.

```bash
# dari CSV frame-level (bisa hitung temporal features lengkap)
python scripts/03_inference_v2.py --input data/au_features/test_au_features.csv

# dari CSV aggregated
python scripts/03_inference_v2.py --input data/au_features/test_au_features_agg.csv --agg

# satu video: masukkan 20 nilai AU langsung
python scripts/03_inference_v2.py --single 0.1 0.0 0.8 ...
```

Output per video: prediksi binary (0/1) dan confidence score untuk tiap state.

---

## Setup

```bash
git clone https://github.com/aitf-sr1/mapping-fer-with-au.git
cd mapping-fer-with-au

python -m venv venv
source venv/bin/activate
pip install py-feat==0.6.2 xgboost scikit-learn scipy pandas tqdm opencv-python
```

**Catatan untuk Python 3.12 + NumPy 2.x:** py-feat 0.6.2 belum update untuk versi terbaru, perlu beberapa patch manual di site-packages:

- `feat/utils/stats.py` — ganti `from scipy.integrate import simps` jadi `simpson`
- `feat/utils/image_operations.py` — ganti `np.mat(` jadi `np.asmatrix(`
- `feat/emo_detectors/ResMaskNet/resmasknet_test.py` — hapus `from lib2to3.pytree import convert`
- `feat/data.py` — tambah recursion guard di `Fex.__init__` untuk pandas 2.x

Detail patch ada di bagian bawah README ini.

---

## Struktur Direktori

```
mapping-fer-with-au/
├── scripts/
│   ├── 01_extract_au_features.py
│   ├── 02_calibrate_v2.py
│   └── 03_inference_v2.py
├── notebooks/
│   └── colab_au_extraction.ipynb    # untuk Colab A100
├── models/
│   └── calibration_v2.json          # threshold optimal per state
├── data/
│   └── au_features/                 # di-ignore git, generate sendiri
└── README.md
```

Dataset DAiSEE tidak disertakan di repo. Struktur folder yang diharapkan:

```
DAiSEE_extracted/
└── DAiSEE/
    ├── DataSet/
    │   ├── Train/
    │   ├── Test/
    │   └── Validation/
    └── Labels/
        ├── TrainLabels.csv
        ├── TestLabels.csv
        └── ValidationLabels.csv
```

---

## Colab A100

Buka `notebooks/colab_au_extraction.ipynb` di Google Colab, ganti runtime ke GPU A100. Script ekstraksi bisa jalan dengan `--batch-size 16` dan estimasi selesai dalam ~2-3 jam untuk semua 8566 video.

Notebook otomatis resume kalau koneksi putus — video yang sudah masuk CSV tidak diproses ulang.

---

## 20 Action Units yang Diekstrak

`AU01` Inner Brow Raise — `AU02` Outer Brow Raise — `AU04` Brow Lowerer — `AU05` Upper Lid Raiser — `AU06` Cheek Raiser — `AU07` Lid Tightener — `AU09` Nose Wrinkler — `AU10` Upper Lip Raiser — `AU11` Nasolabial Deepener — `AU12` Lip Corner Puller — `AU14` Dimpler — `AU15` Lip Corner Depressor — `AU17` Chin Raiser — `AU20` Lip Stretcher — `AU23` Lip Tightener — `AU24` Lip Pressor — `AU25` Lips Part — `AU26` Jaw Drop — `AU28` Lip Suck — `AU43` Eyes Closed

---

## Patch py-feat 0.6.2 untuk Python 3.12

<details>
<summary>Lihat detail patch</summary>

**1. `feat/utils/stats.py`** — scipy rename `simps` → `simpson`:
```python
# ganti:
from scipy.integrate import simps
# jadi:
try:
    from scipy.integrate import simpson as simps
except ImportError:
    from scipy.integrate import simps
```

**2. `feat/utils/image_operations.py`** — NumPy 2.0 hapus `np.mat`:
```bash
sed -i 's/np\.mat(/np.asmatrix(/g' feat/utils/image_operations.py
```

**3. `feat/emo_detectors/ResMaskNet/resmasknet_test.py`** — hapus baris ini (unused import, lib2to3 tidak ada di Python 3.12):
```python
from lib2to3.pytree import convert  # hapus baris ini
```

**4. `feat/data.py`** — recursion guard untuk pandas 2.x (cari blok "Set _metadata attributes on series"):
```python
# ganti:
for k in self:
    self[k].sampling_freq = self.sampling_freq
    self[k].sessions = self.sessions

# jadi:
if not getattr(self, '_in_fex_init', False):
    try:
        object.__setattr__(self, '_in_fex_init', True)
        for k in self:
            self[k].sampling_freq = self.sampling_freq
            self[k].sessions = self.sessions
    finally:
        object.__setattr__(self, '_in_fex_init', False)
```

</details>
