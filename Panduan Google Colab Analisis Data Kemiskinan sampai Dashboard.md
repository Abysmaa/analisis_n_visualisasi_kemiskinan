# Panduan Google Colab: Analisis Data Kemiskinan sampai Dashboard

Panduan ini dibuat setelah dataset Dataset\_Kemiskinan.csv diperiksa lebih dulu. Kode di Langkah 1 sampai 8 sudah dicoba pada dataset tersebut dan berjalan. Langkah dashboard (9 dan 10) memakai pustaka Plotly dan Streamlit yang belum diuji di sini, jadi jalankan dan perbaiki bila ada pesan error.

## Temuan Penting Sebelum Mulai

Dataset ini punya masalah kualitas data yang harus ditangani dulu, kalau tidak hasil analisisnya menyesatkan:

- 16 baris persis kembar, dan 2 pasang kabupaten/kota lain bernama sama tapi nilainya berbeda.
- Kolom Kemiskinan berisi nilai mustahil: Kota Medan bernilai 834 (persen), dan 7 baris bernilai di bawah 1.
- Kolom Klasifikasi Kemiskinan tidak konsisten dengan nilai Kemiskinan: ada baris kelas 1 dengan nilai sekitar 1,01% dan baris kelas 0 dengan nilai sekitar 46,9%. Jadi label ini tidak bisa dipercaya.
- Rata-rata lama sekolah (RLS) ada 148 baris di bawah 4 tahun, dan 33 di antaranya di bawah 3 tahun. Ini tidak masuk akal untuk rata-rata kabupaten/kota.
- Pengeluaran Kota Bogor (117.166) dan Kabupaten Bogor (104.100) jauh di atas wilayah lain, kemungkinan salah skala atau salah ketik. TPT ada yang mendekati 99, dan sanitasi ada yang bernilai 0.
- Akibatnya, korelasi mentah antar variabel terlihat hampir nol (misalnya IPM dengan Kemiskinan hanya sekitar -0,07). Setelah pembersihan, korelasi Spearman IPM dengan Kemiskinan menjadi sekitar -0,47 dan Pengeluaran sekitar -0,49. Jadi pembersihan mengubah kesimpulan.

Tiga pilihan:

1. **Lanjutkan dengan dataset ini** setelah dibersihkan seperti di bawah, dan tulis keterbatasannya dengan jujur di laporan.
2. **Ambil ulang dari BPS** (lebih kuat untuk portofolio karena sumbernya resmi dan bisa diverifikasi).
3. **Gabungan:** gunakan dataset ini untuk berlatih, lalu ulangi pipeline yang sama pada data BPS.

Setelah pembersihan di bawah, sekitar 61 dari 516 baris (sekitar 12%) mengandung nilai yang tidak valid. Itu cukup banyak, jadi pilihan 2 atau 3 disarankan untuk versi final portofolio.

## Langkah 1: Siapkan Colab

Buka colab.research.google.com, buat notebook baru bernama analisis\_kemiskinan, lalu jalankan sel berikut dan unggah file CSV.

```python
import pandas as pd, numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from google.colab import files

files.upload()  # pilih Dataset_Kemiskinan.csv
```

## Langkah 2: Muat dan Periksa Data

```python
df = pd.read_csv('Dataset_Kemiskinan.csv', sep=';', decimal=',')
print(df.shape)
df.info()
print(df.isna().sum())
print(df.duplicated().sum(), 'baris kembar persis')
df.describe().T
```

Pembacaan dengan `decimal=','` langsung mengubah angka berkoma menjadi numerik, jadi langkah konversi manual tidak diperlukan.

## Langkah 3: Hapus Duplikat dan Validasi Nilai

Pakai aturan batas yang masuk akal untuk tiap variabel, lalu hitung berapa nilai yang melanggar.

```python
df = df.drop_duplicates().reset_index(drop=True)
print(df.duplicated(['Provinsi','KabKota']).sum(), 'nama kembar dengan nilai berbeda: periksa manual')

batas = {
    'Kemiskinan': (0, 60), 'RLS': (3, 13), 'Pengeluaran': (4000, 30000),
    'IPM': (45, 90), 'UHH': (55, 80), 'Sanitasi': (5, 100),
    'AirMinum': (5, 100), 'TPT': (0, 15), 'TPAK': (50, 100), 'PDRB': (1, np.inf)
}
anomali = pd.DataFrame({c: df[c].notna() & ~df[c].between(*r) for c, r in batas.items()})
print(anomali.sum())
print('baris dengan minimal satu anomali:', anomali.any(axis=1).sum())
```

Catat tabel jumlah anomali per kolom ini. Tabel itu akan masuk ke laporan sebagai bagian Data Quality.

## Langkah 4: Bersihkan Data

Nilai di luar batas dijadikan kosong (NaN), lalu diisi dengan median provinsi yang sama. Median dipilih karena tidak terpengaruh oleh pencilan, dan provinsi dipakai supaya nilai pengisi sesuai konteks wilayah.

```python
bersih = df.copy()
for c, r in batas.items():
    bersih.loc[~bersih[c].between(*r), c] = np.nan

fitur = list(batas)
for c in fitur:
    bersih[c] = bersih[c].fillna(bersih.groupby('Provinsi')[c].transform('median'))
    bersih[c] = bersih[c].fillna(bersih[c].median())

print(bersih[fitur].isna().sum().sum(), 'nilai kosong tersisa')
```

Catatan: kolom Klasifikasi Kemiskinan tidak dipakai karena tidak konsisten. Kita buat label baru di Langkah 7.

## Langkah 5: Analisis Eksplorasi (EDA)

```python
print(bersih[fitur].describe().round(2))

top10 = bersih.nlargest(10, 'Kemiskinan')[['Provinsi','KabKota','Kemiskinan']]
bawah10 = bersih.nsmallest(10, 'Kemiskinan')[['Provinsi','KabKota','Kemiskinan']]
display(top10, bawah10)

prov = (bersih.groupby('Provinsi')[['Kemiskinan','IPM','Pengeluaran','TPT']]
        .mean().round(2).sort_values('Kemiskinan', ascending=False))
display(prov.head(10))

px.histogram(bersih, x='Kemiskinan', nbins=30, title='Sebaran persentase kemiskinan').show()
px.box(bersih, x='Provinsi', y='Kemiskinan', title='Kemiskinan per provinsi').update_xaxes(tickangle=60).show()
```

Tulis 2 sampai 3 kalimat temuan dari tiap tabel atau grafik. Misalnya provinsi mana yang rata-ratanya tertinggi dan seberapa lebar sebaran antar kabupaten/kota dalam satu provinsi.

## Langkah 6: Korelasi

Gunakan korelasi Spearman karena data kemiskinan tidak berdistribusi normal dan masih punya pencilan.

```python
corr = bersih[fitur].corr(method='spearman').round(2)
fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                           colorscale='RdBu', zmid=0, text=corr.values, texttemplate='%{text}'))
fig.update_layout(title='Korelasi Spearman', height=600)
fig.show()

print(corr['Kemiskinan'].sort_values())

px.scatter(bersih, x='Pengeluaran', y='Kemiskinan', color='Provinsi', hover_name='KabKota',
           trendline='ols', title='Pengeluaran dan kemiskinan').show()
```

Ingat: korelasi menunjukkan hubungan, bukan sebab-akibat. Tulis kalimat itu di kesimpulan.

## Langkah 7: Model Sederhana (Opsional tapi Bernilai)

Buat label baru dari nilai Kemiskinan (kuartil teratas dianggap kemiskinan tinggi), lalu latih model klasifikasi. Kolom Kemiskinan tidak boleh dimasukkan sebagai fitur karena label dibuat darinya (kebocoran data).

```python
bersih['Tinggi'] = (bersih['Kemiskinan'] >= bersih['Kemiskinan'].quantile(.75)).astype(int)

X = bersih[[c for c in fitur if c != 'Kemiskinan']]
y = bersih['Tinggi']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42).fit(Xtr, ytr)
print(classification_report(yte, model.predict(Xte), digits=3))

penting = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
px.bar(penting, title='Variabel paling berpengaruh pada model').show()
```

Pada pengujian di sini, akurasi sekitar 0,85 dengan F1 untuk kelas kemiskinan tinggi sekitar 0,69. Variabel terpenting adalah UHH, Pengeluaran, IPM, dan PDRB, tetapi selisih tingkat pentingnya kecil. Hasil Anda mungkin sedikit berbeda. Jelaskan di laporan bahwa model ini untuk eksplorasi, bukan alat prediksi kebijakan.

## Langkah 8: Simpan Data Bersih

```python
bersih.drop(columns=['Klasifikasi Kemiskinan']).to_csv('data_bersih.csv', index=False)
files.download('data_bersih.csv')
```

File ini bisa langsung dipakai di Power BI atau Looker Studio kalau mau membuat dashboard di sana.

## Langkah 9: Dashboard Interaktif dengan Plotly (Satu File HTML)

```python
top = bersih.nlargest(10, 'Kemiskinan').sort_values('Kemiskinan')
prov_rata = bersih.groupby('Provinsi')['Kemiskinan'].mean().sort_values().tail(10)

dash = make_subplots(rows=2, cols=2,
    subplot_titles=('10 kabupaten/kota dengan kemiskinan tertinggi', 'Rata-rata kemiskinan per provinsi (10 tertinggi)',
                    'Pengeluaran dan kemiskinan', 'Korelasi dengan kemiskinan'))
dash.add_trace(go.Bar(x=top['Kemiskinan'], y=top['KabKota'], orientation='h', name='Kabupaten/Kota'), row=1, col=1)
dash.add_trace(go.Bar(x=prov_rata.values, y=prov_rata.index, orientation='h', name='Provinsi'), row=1, col=2)
dash.add_trace(go.Scatter(x=bersih['Pengeluaran'], y=bersih['Kemiskinan'], mode='markers',
                          text=bersih['KabKota'], name='Wilayah'), row=2, col=1)
kor = corr['Kemiskinan'].drop('Kemiskinan').sort_values()
dash.add_trace(go.Bar(x=kor.values, y=kor.index, orientation='h', name='Korelasi'), row=2, col=2)
dash.update_layout(height=850, showlegend=False, title_text='Dashboard Analisis Kemiskinan Kabupaten/Kota')

dash.write_html('dashboard_kemiskinan.html', include_plotlyjs='cdn')
files.download('dashboard_kemiskinan.html')
```

File HTML ini bisa dibuka di browser dan dipublikasikan lewat GitHub Pages atau web portofolio.

## Langkah 10: Dashboard Streamlit (Opsional)

Streamlit tidak nyaman dijalankan langsung di Colab. Jalankan di komputer sendiri atau unggah ke GitHub lalu deploy di Streamlit Community Cloud. Simpan sebagai app.py, di folder yang sama dengan data\_bersih.csv, dan buat requirements.txt berisi streamlit, pandas, plotly.

```python
import pandas as pd, plotly.express as px, streamlit as st

st.set_page_config(page_title='Dashboard Kemiskinan', layout='wide')
df = pd.read_csv('data_bersih.csv')

st.title('Dashboard Analisis Kemiskinan Kabupaten/Kota')
pilih = st.sidebar.multiselect('Provinsi', sorted(df['Provinsi'].unique()))
d = df[df['Provinsi'].isin(pilih)] if pilih else df

k1, k2, k3 = st.columns(3)
k1.metric('Jumlah wilayah', len(d))
k2.metric('Rata-rata kemiskinan (%)', round(d['Kemiskinan'].mean(), 2))
k3.metric('Rata-rata IPM', round(d['IPM'].mean(), 2))

c1, c2 = st.columns(2)
c1.plotly_chart(px.bar(d.nlargest(10, 'Kemiskinan'), x='Kemiskinan', y='KabKota', orientation='h',
                       title='10 tertinggi'), use_container_width=True)
c2.plotly_chart(px.scatter(d, x='Pengeluaran', y='Kemiskinan', color='Provinsi', hover_name='KabKota',
                           title='Pengeluaran dan kemiskinan'), use_container_width=True)
st.dataframe(d)
```

Jalankan dengan `streamlit run app.py`.

## Langkah 11: Laporan dan Portofolio

Susun repositori GitHub seperti ini:

```
analisis-kemiskinan/
  data/raw/Dataset_Kemiskinan.csv
  data/clean/data_bersih.csv
  notebook/analisis_kemiskinan.ipynb
  dashboard/dashboard_kemiskinan.html
  README.md
```

Isi README: masalah yang ingin dijawab, sumber data, temuan kualitas data (tabel dari Langkah 3), langkah pembersihan beserta alasannya, 3 sampai 4 temuan utama, keterbatasan, dan tautan dashboard. Bagian keterbatasan penting: sebutkan bahwa sebagian nilai tidak valid dan diisi median provinsi, dan bahwa korelasi bukan sebab-akibat.

Ringkasan satu halaman untuk rekruter: pertanyaan, 3 temuan dengan angka, 1 grafik terbaik, dan 1 rekomendasi.

## Daftar Periksa

- [ ] Notebook berjalan dari atas ke bawah tanpa error (Runtime, lalu Restart and run all).
- [ ] Tabel anomali tercatat dan dijelaskan.
- [ ] Setiap grafik punya judul dan satu kalimat temuan.
- [ ] Label Klasifikasi Kemiskinan lama tidak dipakai, dan alasannya tertulis.
- [ ] Dashboard HTML bisa dibuka di ponsel.
- [ ] README ada di GitHub, lalu tautannya dipasang di web portofolio.
