# Analisis Sentimen Opini Publik Mengenai Program Makan Bergizi Gratis Menggunakan BERT-CNN

**Peneliti:** Khoeru Roziqin (24060119120031)
**Program Studi:** S1 Informatika, Fakultas Sains dan Matematika, Universitas Diponegoro
**Skripsi:** Analisis Sentimen Opini Publik di Platform Media Sosial X Mengenai Program Makan Bergizi Gratis Menggunakan BERT-CNN

---

## Ringkasan

Repositori ini berisi pipeline penelitian lengkap untuk klasifikasi sentimen opini publik di platform X (Twitter) mengenai Program Makan Bergizi Gratis (MBG). Model yang diusulkan adalah arsitektur Dual-Path yang menggabungkan IndoBERT dengan Convolutional Neural Network (CNN) 1D untuk mengklasifikasikan tweet ke dalam tiga kelas sentimen: positif, negatif, dan netral.

Pipeline penelitian meliputi crawling data, preprocessing berbasis Transformer, pelabelan otomatis sebagai validasi Inter-Annotator Agreement (IAA), dan training model dengan optimasi hyperparameter tiga fase. Seluruh proses disajikan sebagai empat notebook Jupyter berurutan.

- **Task:** Klasifikasi sentimen 3 kelas (positif / negatif / netral)
- **Bahasa:** Bahasa Indonesia (`lang:id`)
- **Model dasar:** IndoBERT (`indobenchmark/indobert-base-p2`) + CNN 1D Dual-Path

---

## Alur Penelitian

```
00_crawling         Crawling tweet dari X via tweet-harvest
                    keyword: "makan bergizi gratis" + "mbg"
                    rentang: 6 Januari 2025 - 6 Januari 2026
                                        |
01_data_preparation Merge, deduplikasi berlapis, filter relevansi,
                    stratified sampling per periode
                                        |
02_preprocessing    Preprocessing 4 tahap untuk BERT,
   & labeling       pelabelan otomatis IndoRoBERTa,
                    anotasi manual, dan Cohen's Kappa (IAA)
                                        |
03_training         IndoBERT + Dual-Path CNN,
                    grid search 3 fase, 5-fold Cross-Validation,
                    evaluasi test set
```

Setiap tahap dibuat sebagai notebook mandiri di `research/pipeline/`, dengan seluruh figur dan metrik disimpan pada folder terkait di `research/results/`.

---

## Struktur Repositori

```
mbg_bert_cnn/
├── dataset/
│   ├── raw/                              # hasil crawling mentah (git-ignored)
│   │   ├── makan_bergizi_gratis/
│   │   └── mbg/
│   ├── merge/                            # data merge dan validasi (git-ignored)
│   └── final/
│       ├── raw_sampling_mbg.csv              # sampel tweet pra-anotasi
│       ├── raw_sample_labeled_mbg.csv        # working file anotasi manual
│       ├── final_validated_mbg.csv           # label tervalidasi + metadata
│       └── final_mbg_labeled.csv             # ★ korpus final berlabel
│
├── preprocessing/kamus/                  # kamus custom domain MBG
│   ├── kamus_alay_mbg.csv
│   ├── demoji_code_mbg.csv
│   ├── akun_x_mbg.csv
│   ├── whitelist_hashtag_mbg.csv
│   └── additional_stopwords_mbg.csv
│
├── research/
│   ├── pipeline/                         # ★ empat notebook pipeline utama
│   │   ├── 00_crawling_data_x.ipynb
│   │   ├── 01_data_preparation.ipynb
│   │   ├── 02_preprocessing_and_labeling.ipynb
│   │   └── 03_indobert_cnn_training.ipynb
│   └── results/
│       ├── 01_data_preparation/              # tren temporal, distribusi sampling
│       ├── 02_preprocessing_labeling/        # distribusi confidence, IAA
│       └── 03_model_results/                 # log, metrik CV, model checkpoint
│
├── LICENSE
└── README.md
```

> File CSV mentah, hasil merge, dan checkpoint model (`*.pt`) dikecualikan dari version control melalui `.gitignore` karena keterbatasan ukuran dan ketentuan platform. Korpus final berlabel dan seluruh artefak hasil (metrik, log, figur) tetap dilacak.

---

## Arsitektur Model

Model yang diusulkan menggabungkan dua representasi komplementer dari `last_hidden_state` IndoBERT:

- **Path 1 (Token [CLS]):** representasi konteks global kalimat hasil agregasi self-attention 12 layer IndoBERT
- **Path 2 (CNN 1D Multi-Kernel):** pola n-gram lokal melalui Conv1D paralel dengan berbagai ukuran window, diikuti Global Max Pooling

Kedua path di-concatenate sebelum classifier head (Dense 256 -> Dense 3). CNN berperan melengkapi [CLS] daripada menggantikannya.

Training menggunakan differential learning rate (LR kecil untuk IndoBERT pre-trained, LR lebih besar untuk CNN dan classifier head yang dilatih dari nol) dan weight decay exclusion pada parameter `bias` serta `LayerNorm`.

---

## Ringkasan Hasil

Evaluasi pada test set fixed 1.329 tweet (20% dari korpus berlabel), dievaluasi sekali dengan model Dual-Path pada kondisi S2 (Random Undersampling):

| Metrik            | Skor   |
| ----------------- | ------ |
| Accuracy          | 85,70% |
| F1-Macro          | 0,8547 |
| F1-Weighted       | 0,8587 |
| Precision (Macro) | 0,8551 |
| Recall (Macro)    | 0,8585 |

Per-kelas F1: `positive` = 0,889 | `negative` = 0,872 | `neutral` = 0,803

5-fold Cross-Validation pada train+val (S2): F1-Macro = 0,846 ± 0,016

Confusion matrix, learning curves, sensitivity plot LR, dan figur stabilitas K-Fold tersimpan di [`research/results/03_model_results/`](research/results/03_model_results/).

---

## Dataset

Tweet dikumpulkan melalui [`tweet-harvest`](https://github.com/helmisatria/tweet-harvest) v2.7.1 menggunakan dua keyword utama: `makan bergizi gratis` (nama program formal) dan `mbg` (akronim informal), dibatasi `lang:id` pada tab LATEST, selama satu tahun pelaksanaan program.

| Tahap                      | Jumlah Tweet |
| -------------------------- | -----------: |
| Crawling mentah            |      172.009 |
| Setelah merge dan validasi |      118.340 |
| Setelah dedup dan filter   |      ~53.669 |
| Hasil stratified sampling  |        6.840 |
| **Korpus final berlabel**  |    **6.642** |

Sampling menggunakan pendekatan equal-allocation stratified per 12 periode bulanan (basis tanggal 6 setiap bulan) dengan 570 tweet per periode (500 target statistik + 70 buffer kurasi). Pendekatan ini menjamin representasi temporal yang merata sepanjang tahun.

Distribusi label pada korpus final: positive (39,5%), negative (31,4%), neutral (29,1%). Korpus di-split 80/20 secara stratified menjadi train+val (5.313 tweet) dan test set fixed (1.329 tweet).

Ground truth berasal dari anotasi manual peneliti. IndoRoBERTa (`w11wo/indonesian-roberta-base-sentiment-classifier`) berperan sebagai pseudo-annotator kedua untuk perhitungan Inter-Annotator Agreement dengan hasil Cohen's Kappa = 0,644 (kategori substantial agreement, Landis dan Koch, 1977).

---

## Pernyataan Etika dan Data

- Data berasal dari postingan publik di platform X, dikumpulkan semata-mata untuk kepentingan penelitian akademik non-komersial sesuai ketentuan platform.
- Analisis dilakukan secara agregat, tanpa upaya mengidentifikasi atau memprofilkan pengguna individual.
- Label otomatis IndoRoBERTa bersifat pseudo-annotator untuk perhitungan IAA. Ground truth penelitian tetap merujuk pada anotasi manual peneliti.
- Analisis sentimen menghasilkan gambaran eksploratoris yang terikat pada cakupan metodologis penelitian, bukan representasi definitif opini publik keseluruhan.
- Konten tweet mentah tidak didistribusikan kembali dan dikecualikan dari version control. Hanya artefak turunan dan berlabel yang dibagikan.

---

## Sitasi

Jika menggunakan repositori ini, mohon mengutip:

```bibtex
@thesis{roziqin2026mbgbertcnn,
  author = {Khoeru Roziqin},
  title  = {Analisis Sentimen Opini Publik di Platform Media Sosial X
            Mengenai Program Makan Bergizi Gratis Menggunakan BERT-CNN},
  school = {Universitas Diponegoro},
  year   = {2026},
  type   = {Skripsi Sarjana}
}
```

Model pre-trained yang digunakan:

- [`indobenchmark/indobert-base-p2`](https://huggingface.co/indobenchmark/indobert-base-p2) (encoder utama)
- [`w11wo/indonesian-roberta-base-sentiment-classifier`](https://huggingface.co/w11wo/indonesian-roberta-base-sentiment-classifier) (pseudo-annotator IAA)

---

## Lisensi

Dirilis ke domain publik di bawah [The Unlicense](LICENSE). Data tweet dan model pre-trained pihak ketiga tetap tunduk pada ketentuan platform dan lisensi model masing-masing.
