# Tutorial Penggunaan

Panduan reproduksi penelitian dari crawling data hingga evaluasi model. Dokumen ini fokus pada langkah eksekusi. Untuk detail arsitektur, metrik, dan justifikasi metodologis, lihat [`README.md`](README.md).

---

## Prasyarat

Akun dan tool yang perlu disiapkan:

- **Akun Google** untuk Google Colab dan Google Drive (tahap crawling)
- **Akun Kaggle** untuk Kaggle Notebook (tahap preparation, preprocessing, training)
- **Akun X (Twitter)** untuk mendapatkan `auth_token` (tahap crawling)
- **Git** (opsional) untuk clone repositori

Perkiraan waktu total: sekitar 28 jam mesin GPU T4 untuk training model, ditambah waktu crawling dan anotasi manual (bervariasi).

---

## Setup Awal (Satu Kali)

### 1. Clone atau download repositori

Repositori tersedia melalui dua sumber:

**GitHub (utama):**

```bash
git clone https://github.com/roziqinkhoeru/mbg_bertn_cnn.git
cd mbg_bertn_cnn
```

**Google Drive (alternatif dengan isi paling lengkap):**

<https://drive.google.com/drive/folders/1t7SyuepCoS5djYdkbmraAitj2y0-wDka?usp=sharing>

> Google Drive menyimpan **seluruh artefak penelitian** termasuk dataset mentah hasil crawling, dataset merge, hasil eksperimen tiap fase, serta checkpoint model. File-file ini dikecualikan dari GitHub karena batasan ukuran dan ketentuan platform. Gunakan Drive apabila membutuhkan data lengkap peneliti untuk verifikasi atau ingin melewati tahap crawling.

### 2. Upload kamus ke Kaggle

Buat Kaggle Dataset baru dengan slug `mbg-kamus` dan upload seluruh isi folder `preprocessing/kamus/`:

- `kamus_alay_mbg.csv`
- `demoji_code_mbg.csv`
- `akun_x_mbg.csv`
- `whitelist_hashtag_mbg.csv`
- `additional_stopwords_mbg.csv`

Kamus ini akan menjadi input untuk NB02.

---

## Tahap 1: Crawling Data (NB00, Google Colab)

> **Alternatif tanpa crawling:** Dataset mentah lengkap hasil crawling peneliti tersedia pada Google Drive (folder `dataset/raw/`). Jika ingin melewati tahap ini, download folder tersebut kemudian lanjut ke bagian "Persiapan untuk tahap berikutnya" untuk mengupload sebagai Kaggle Dataset.

Notebook ini menggunakan library [`tweet-harvest`](https://github.com/helmisatria/tweet-harvest) v2.7.1 oleh Helmi Satria. Untuk penjelasan mekanisme crawling dan cara mendapatkan `auth_token` X secara lengkap, silakan merujuk pada artikel resmi author:

> <https://helmisatria.com/blog/cara-crawl-mendapatkan-data-twitter-dengan-filter-waktu-dan-lainnya>

### Langkah eksekusi

1. Buka [Google Colab](https://colab.research.google.com/) dan upload `research/pipeline/00_crawling_data_x.ipynb`.
2. Ambil `auth_token` dari cookie akun X (DevTools > Application > Cookies > `auth_token`).
3. Isi sel Konfigurasi:
   - `TWITTER_AUTH_TOKEN`: token dari langkah 2
   - `KEYWORD_CATEGORY`: `mbg` atau `makan_bergizi_gratis`
   - `SEARCH_QUERY`: contoh `mbg since:2025-11-06 until:2025-11-09 lang:id`
   - `FILENAME`: contoh `6-8-nov-25.csv`
   - `LIMIT`: contoh `5000`
4. Jalankan seluruh sel. Output tersimpan pada Drive di `MyDrive/mbg_bert_cnn/dataset/raw/{kategori}/`.
5. **Ulangi** langkah 3 dan 4 untuk setiap rentang tanggal dan kategori keyword sampai seluruh periode ter-crawl.

### Persiapan untuk tahap berikutnya

- Download seluruh isi folder `MyDrive/mbg_bert_cnn/dataset/raw/` ke lokal (atau ambil dari Google Drive peneliti sesuai alternatif di atas).
- Buat Kaggle Dataset baru dengan slug `mbg-raw-tweets` dengan struktur folder:

  ```
  mbg-raw-tweets/
  ├── makan_bergizi_gratis/
  │   └── *.csv
  └── mbg/
      └── *.csv
  ```

---

## Tahap 2: Data Preparation (NB01, Kaggle)

### Langkah eksekusi

1. Buat Kaggle Notebook baru dan upload `research/pipeline/01_data_preparation.ipynb`.
2. Attach Kaggle Dataset `mbg-raw-tweets` sebagai input.
3. Jalankan seluruh sel (Run All).
4. Output pada `/kaggle/working/dataset/final/raw_sampling_mbg.csv` (sekitar 6.840 tweet).
5. Download file `raw_sampling_mbg.csv` untuk proses anotasi manual di tahap berikutnya.

---

## Tahap 3: Anotasi Manual

Isi kolom `label` untuk setiap baris pada `raw_sampling_mbg.csv` dengan salah satu nilai berikut (**huruf kecil**):

- `positive`
- `negative`
- `neutral`

Tools anotasi bebas (Excel, Google Sheets, Label Studio, doccano, dan sejenisnya).

### Catatan bagi pengguna Label Studio

Ekspor Label Studio menghasilkan kolom `sentiment` dengan nilai sentence case (`Positive`, `Negative`, `Neutral`). Sebelum upload ke Kaggle, sesuaikan agar konsisten dengan format yang dibutuhkan NB02:

- Rename kolom `sentiment` menjadi `label`
- Konversi nilai ke huruf kecil

Contoh dengan pandas:

```python
import pandas as pd

df = pd.read_csv("label_studio_export.csv")
df = df.rename(columns={"sentiment": "label"})
df["label"] = df["label"].str.lower()
df.to_csv("raw_sample_labeled_mbg.csv", index=False)
```

### Simpan dan upload

- Save hasil anotasi sebagai `raw_sample_labeled_mbg.csv`.
- Buat Kaggle Dataset baru dengan slug `mbg-labeled` berisi file ini (versi 1).

---

## Tahap 4: Preprocessing dan Pelabelan (NB02, Kaggle)

### Prasyarat Kaggle Notebook

- **Settings > Accelerator**: GPU T4 (atau setara)
- **Settings > Internet**: On (untuk download IndoRoBERTa dan kamus Nasal)
- **Input datasets**: `mbg-labeled` (versi 1) + `mbg-kamus`

### Langkah eksekusi

1. Upload `research/pipeline/02_preprocessing_and_labeling.ipynb` ke Kaggle Notebook.
2. Jalankan seluruh sel (Run All).
3. Verifikasi hasil Cohen's Kappa pada sel IAA. Target minimum: 0,61 (substantial agreement).
4. Output pada `/kaggle/working/dataset/final/`:
   - `final_validated_mbg.csv` (audit trail lengkap)
   - `final_mbg_labeled.csv` (input untuk NB03)

### Update Kaggle Dataset `mbg-labeled`

- Download `final_mbg_labeled.csv`.
- Buat versi baru Kaggle Dataset `mbg-labeled` yang berisi kedua file:

  ```
  mbg-labeled/
  ├── raw_sample_labeled_mbg.csv
  └── final_mbg_labeled.csv
  ```

---

## Tahap 5: Training Model (NB03, Kaggle Multi-Sesi)

Training dijalankan dalam **4 sesi** karena batas eksekusi Kaggle (9 jam per sesi) sedangkan total training memerlukan sekitar 28 jam GPU.

### Prasyarat setiap sesi

- **Settings > Accelerator**: GPU T4
- **Settings > Internet**: On
- **Input datasets**: `mbg-labeled` (versi terbaru) + `mbg-training-outputs` (mulai sesi 2)
- Upload `research/pipeline/03_indobert_cnn_training.ipynb`

### Mekanisme umum antar-sesi

Setiap sesi dimulai dengan menjalankan sel Setup, Konfigurasi, Load Data, Model, dan Training Utilities. Kaggle akan mereset `/kaggle/working/` setiap sesi, sehingga output sesi sebelumnya harus di-restore dari Kaggle Dataset `mbg-training-outputs`. Restore ini dilakukan otomatis oleh sel Konfigurasi selama dataset tersebut sudah di-attach.

### Sesi 1: Phase 1 (Base Training Parameter)

Belum ada dataset `mbg-training-outputs` di sesi ini. Attach `mbg-labeled` saja.

- Jalankan sel Setup, Konfigurasi, Load, Model, Utilities.
- Jalankan sel Phase 1 (combo, execution, visualisasi).
- Perkiraan waktu: 6-8 jam.
- Setelah selesai:
  1. Download folder `research/results/03_model_results/` dari `/kaggle/working/`.
  2. Buat Kaggle Dataset baru dengan slug `mbg-training-outputs` berisi seluruh file di folder tersebut (versi 1).
  3. **Catat konfigurasi terbaik** dari output Phase 1 (`lr_bert`, `lr_cnn`, `batch_size`).

### Sesi 2: Phase 2A (CNN Config, ngram [1,2,3])

- Update variabel `BEST_LR_BERT`, `BEST_LR_CNN`, `BEST_BS` di sel Konfigurasi dengan hasil Phase 1.
- Attach `mbg-labeled` dan `mbg-training-outputs` (versi 1).
- Jalankan sel Setup sampai Utilities. File Phase 1 otomatis ter-restore.
- **Skip** sel Phase 1 (lihat catatan skip di bawah).
- Jalankan sel Phase 2A (combo dan execution).
- Perkiraan waktu: 6-8 jam.
- Setelah selesai, update `mbg-training-outputs` (versi 2) dengan seluruh isi folder `03_model_results/`.

### Sesi 3: Phase 2B (CNN Config, ngram [2,3,4]) + Combined Viz

- Konfigurasi `BEST_*` sama dengan Sesi 2.
- Attach `mbg-training-outputs` (versi 2).
- Skip sel Phase 1 dan Phase 2A.
- Jalankan sel Phase 2B (combo dan execution) dan sel Phase 2 Combined Viz.
- Perkiraan waktu: 6-8 jam.
- Setelah selesai, **catat konfigurasi CNN terbaik** dari output combined (`ngram`, `filter`, `dropout`, `activation`), lalu update `mbg-training-outputs` (versi 3).

### Sesi 4: Phase 3 K-Fold dan Final Training

- Update variabel `BEST_NGRAM`, `BEST_FILTER`, `BEST_DROPOUT`, `BEST_ACTIVATION` dengan hasil Phase 2.
- Attach `mbg-training-outputs` (versi 3).
- Skip Phase 1, Phase 2A, dan Phase 2B.
- Jalankan sel Phase 3 (assert, K-Fold, viz) dan Final Training.
- Perkiraan waktu: 3-4 jam.

Output final tersimpan pada:

- `research/results/03_model_results/saved_models/indobert_cnn_dualpath_S2.pt` (checkpoint model)
- `research/results/03_model_results/test_predictions.csv` (prediksi per tweet)
- `research/results/03_model_results/proposed_S2_cm_pair.png` (confusion matrix)
- Metrik final tercetak pada output sel

### Cara skip sel di Kaggle Notebook

Dua opsi:

1. **Eksekusi manual per sel (rekomendasi)**: klik tombol Run pada sel yang ingin dijalankan saja, tidak perlu memodifikasi kode. Cocok kalau tidak ingin menyentuh kode notebook.
2. **Restart & Run All**: bungkus sel execution fase yang di-skip dengan `if False:` block, contohnya:

   ```python
   if False:
       # ... isi sel Phase 1 execution ...
       pass
   ```

   Setelah tidak diperlukan lagi (di sesi berikutnya), sel dapat dikembalikan seperti semula.

| ---                                               | Isi                                                         |
| ------------------------------------------------- | ----------------------------------------------------------- |
| `dataset/final/final_mbg_labeled.csv`             | Korpus final berlabel (sekitar 6.642 tweet)                 |
| `dataset/final/final_validated_mbg.csv`           | Audit trail dengan label manual + IndoRoBERTa + confidence  |
| `research/results/01_data_preparation/`           | Chart tren temporal, distribusi sampling, EDA panjang token |
| `research/results/02_preprocessing_labeling/`     | Chart distribusi confidence dan hasil IAA                   |
| `research/results/03_model_results/`              | Metrik CSV per fase, PNG visualisasi, prediksi test set     |
| `research/results/03_model_results/saved_models/` | Model checkpoint `.pt` + IndoRoBERTa + confidence           |
| `research/results/01_data_preparation/`           | Chart tren temporal, distribusi sampling, EDA panjang token |
| `research/results/02_preprocessing_labeling/`     | Chart distribusi confidence dan hasil IAA                   |
| `research/results/03_model_results/`              | Metrik CSV per fase, PNG visualisasi, prediksi test set     |
| `research/results/03_model_results/saved_models/` | Model checkpoint `.pt`                                      |

> Seluruh output final peneliti juga tersedia pada Google Drive (link di Setup Awal), termasuk model checkpoint terlatih apabila ingin langsung digunakan tanpa training ulang.

---

## Troubleshooting

**Crawling mengembalikan 0 tweet**
`auth_token` X kemungkinan telah expired. Refresh cookie dari browser dan set ulang variabel `TWITTER_AUTH_TOKEN` di NB00.

**Terkena rate limit X saat crawling**
Tunggu 15 sampai 30 menit sebelum mencoba kembali. Kurangi `LIMIT` per sesi jika sering terkena.

**Cohen's Kappa NB02 di bawah 0,61**
Review ulang tweet dengan `confidence_roberta < 0.60` (tersaji di distribusi confidence). Perbaiki anotasi manual yang meragukan pada baris tersebut lalu jalankan ulang NB02.

**Kaggle GPU quota habis**
Batas free tier sekitar 30 jam GPU per minggu. Tunggu reset mingguan atau lanjutkan sesi berikutnya di minggu depan.

**File tidak ter-restore di NB03 sesi 2 ke atas**
Pastikan slug Kaggle Dataset input persis bernama `mbg-training-outputs` dan versi terbaru sudah di-attach. Cek juga output sel Konfigurasi. Jika tidak muncul "Restoring output sesi sebelumnya", berarti dataset belum terdeteksi.

**Sel `assert` gagal di awal Phase 2 atau Phase 3**
Variabel `BEST_*` yang direferensikan belum diupdate dari output fase sebelumnya. Pesan error akan menyebutkan variabel yang bermasalah, buka sel Konfigurasi dan isi nilainya sesuai output fase sebelumnya.
