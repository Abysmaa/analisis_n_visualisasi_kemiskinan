# Analisis Kemiskinan Kabupaten/Kota

Analisis ini menjawab: faktor apa yang paling berkaitan dengan kemiskinan, wilayah mana yang memiliki tingkat kemiskinan tertinggi, dan apakah pola tersebut dapat dimodelkan secara eksploratif.

## Hasil Utama

- Dataset mentah berukuran **532 baris × 13 kolom** dan memiliki **16 baris duplikat persis**.
- Setelah penghapusan duplikat, **61 baris** mengandung setidaknya satu nilai yang melanggar batas yang ditentukan; ini sekitar **11.8%** dari data duplikat-telah-dihapus.
- Korelasi Spearman Pengeluaran–Kemiskinan adalah **-0.49** dan IPM–Kemiskinan adalah **-0.47**.
- Model Random Forest eksploratif memperoleh akurasi **0.846** dan F1 kelas kemiskinan tinggi **0.692**.

## Dashboard

- [Dashboard HTML interaktif](dashboard/dashboard_kemiskinan.html)
- [Notebook](notebook/analisis_kemiskinan.ipynb)
- [Aplikasi Streamlit](dashboard/app.py)

Dashboard memakai palet retro kertas-krem, hijau zaitun, terracotta, dan mustard; judul serif, label monospace, bingkai tebal, dan kartu bersiluet asimetris. Ringkasan visualisasinya disusun sebagai satu papan komposit responsif—bukan grafik-grafik besar yang berdiri sendiri—dengan tujuh panel: peringkat enam wilayah tertinggi, histogram sebaran, donut posisi terhadap median, kurva distribusi kumulatif, scatter pengeluaran–kemiskinan, box plot lima provinsi dengan median tertinggi, serta heatmap korelasi Spearman. Arahkan kursor untuk melihat nilai rinci; pada HTML dan Streamlit, tombol kamera grafik dapat mengunduh satu PNG komposit dengan ukuran dasar 1800 × 1500 dan skala ekspor 2 untuk sampul portofolio. Kurva kumulatif mengurutkan wilayah dan bukan tren waktu; donut adalah pembagian deskriptif, bukan klasifikasi resmi. Versi Streamlit memiliki filter provinsi yang memperbarui semua panel, serta tab terpisah untuk korelasi dan tabel data. Tema terang dikunci di `dashboard/.streamlit/config.toml` agar warna tabel dan kontrol tetap konsisten.

## Struktur Repositori

```text
analisis-kemiskinan/
  data/raw/Dataset_Kemiskinan.csv
  data/clean/data_bersih.csv
  notebook/analisis_kemiskinan.ipynb
  dashboard/dashboard_kemiskinan.html
  dashboard/app.py
  dashboard/requirements.txt
  dashboard/.streamlit/config.toml
  README.md
```

## Kualitas dan Pembersihan Data

Nilai duplikat dihapus, kolom lama `Klasifikasi Kemiskinan` dibuang karena tidak konsisten dengan nilai kemiskinan, dan nilai yang berada di luar rentang wajar dijadikan kosong lalu diisi median Provinsi.

| Variabel | Jumlah nilai anomali |
|---|---:|
| RLS | 33 |
| TPT | 26 |
| Sanitasi | 9 |
| Pengeluaran | 3 |
| AirMinum | 2 |
| IPM | 2 |
| Kemiskinan | 1 |

## Temuan Wilayah

| Provinsi | Kabupaten/Kota | Kemiskinan |
|---|---|---:|
| SULAWESI TENGAH | Morowali | 59.38% |
| JAWA TIMUR | Situbondo | 54.38% |
| JAMBI | Tanjung Jabung Barat | 46.88% |
| SUMATERA SELATAN | Banyu Asin | 46.88% |
| KALIMANTAN TIMUR | Kutai Timur | 43.12% |
| RIAU | Pelalawan | 41.88% |
| PAPUA | Intan Jaya | 41.66% |
| PAPUA | Deiyai | 40.59% |
| JAMBI | Sarolangun | 39.38% |
| PAPUA | Lanny Jaya | 38.73% |

## Korelasi dan Interpretasi

Spearman dipilih karena distribusi kemiskinan tidak normal dan beberapa variabel masih mengandung pencilan. Heatmap korelasi tersedia di notebook serta dashboard. Nilai mendekati -1 menunjukkan hubungan monoton berlawanan, nilai dekat +1 menunjukkan hubungan searah, dan nilai dekat 0 menunjukkan hubungan monoton yang lemah. Korelasi menunjukkan hubungan, bukan sebab-akibat.

## Model Eksploratif

Label dibuat dari kuartil tertinggi kemiskinan. Kolom kemiskinan tidak dimasukkan sebagai fitur.

| Variabel | Tingkat pentingan |
|---|---:|
| UHH | 0.150 |
| Pengeluaran | 0.148 |
| IPM | 0.143 |
| PDRB | 0.125 |
| AirMinum | 0.101 |

Model ini menunjukkan pola dalam data dan tidak dirancang sebagai alat prediksi kebijakan.

## Cara Menjalankan

### Notebook

1. Buka `notebook/analisis_kemiskinan.ipynb` di Google Colab atau Jupyter.
2. Jalankan semua sel dari atas ke bawah.

### Dashboard HTML

1. Unduh atau buka `dashboard/dashboard_kemiskinan.html` dengan browser.
2. Untuk publikasi, unggah repository ke GitHub Pages atau web portofolio.

### Streamlit

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Keterbatasan

- Sebagian nilai berasal dari skala atau format yang tampaknya tidak valid.
- Nilai anomali diimputasi dengan median Provinsi; imputasi dapat memperbesar atau mengurangi variasi lokal.
- Dataset perlu diverifikasi ulang terhadap sumber BPS untuk analisis dan portofolio final.
- Klasifikasi kemiskinan lama tidak digunakan karena tidak konsisten.

## Sumber

File sumber: `data/raw/Dataset_Kemiskinan.csv` (disalin dari `Dataset_Kemiskinan (1).csv`).
