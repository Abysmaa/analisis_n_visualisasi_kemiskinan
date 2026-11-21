from pathlib import Path
import shutil

import nbformat as nbf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Dataset_Kemiskinan (1).csv"

for directory in (
    "data/raw",
    "data/clean",
    "notebook",
    "dashboard",
    "dashboard/.streamlit",
):
    (ROOT / directory).mkdir(parents=True, exist_ok=True)

shutil.copy2(SOURCE, ROOT / "data/raw/Dataset_Kemiskinan.csv")

df = pd.read_csv(SOURCE, sep=";", decimal=",")
raw_shape = df.shape
exact_duplicates = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

batas = {
    "Kemiskinan": (0, 60),
    "RLS": (3, 13),
    "Pengeluaran": (4000, 30000),
    "IPM": (45, 90),
    "UHH": (55, 80),
    "Sanitasi": (5, 100),
    "AirMinum": (5, 100),
    "TPT": (0, 15),
    "TPAK": (50, 100),
    "PDRB": (1, np.inf),
}

anomali = pd.DataFrame(
    {
        c: df[c].notna() & ~df[c].between(*r)
        for c, r in batas.items()
    }
)
anomali_counts = anomali.sum().sort_values(ascending=False)
rows_with_anomaly = int(anomali.any(axis=1).sum())

bersih = df.copy()
for c, r in batas.items():
    bersih.loc[~bersih[c].between(*r), c] = np.nan

fitur = list(batas)
for c in fitur:
    bersih[c] = bersih[c].fillna(
        bersih.groupby("Provinsi")[c].transform("median")
    )
    bersih[c] = bersih[c].fillna(bersih[c].median())

bersih["Tinggi"] = (
    bersih["Kemiskinan"] >= bersih["Kemiskinan"].quantile(0.75)
).astype(int)
bersih.drop(columns=["Klasifikasi Kemiskinan"]).to_csv(
    ROOT / "data/clean/data_bersih.csv",
    index=False,
)

corr = bersih[fitur].corr(method="spearman").round(2)
provinsi = (
    bersih.groupby("Provinsi")[["Kemiskinan", "IPM", "Pengeluaran", "TPT"]]
    .mean()
    .round(2)
)
top_provinsi = provinsi.sort_values("Kemiskinan", ascending=False)
top_area = bersih.nlargest(10, "Kemiskinan")[["Provinsi", "KabKota", "Kemiskinan"]]

X = bersih[[c for c in fitur if c != "Kemiskinan"]]
y = bersih["Tinggi"]
Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
)
model.fit(Xtr, ytr)
report = classification_report(yte, model.predict(Xte), output_dict=True)
accuracy = report["accuracy"]
f1_tinggi = report["1"]["f1-score"]
penting = (
    pd.Series(model.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
)

kor = corr["Kemiskinan"].drop("Kemiskinan").sort_values()
median_poverty = bersih["Kemiskinan"].median()
province_order = (
    bersih.groupby("Provinsi")["Kemiskinan"]
    .median()
    .sort_values()
    .tail(5)
    .index
    .tolist()
)
negative_feature = kor.index[0]
positive_feature = kor.index[-1]
mean_poverty = bersih["Kemiskinan"].mean()
top_location = bersih.loc[bersih["Kemiskinan"].idxmax()]
top = bersih.nlargest(6, "Kemiskinan").sort_values("Kemiskinan")
sorted_poverty = bersih["Kemiskinan"].sort_values().reset_index(drop=True)
overview = make_subplots(
    rows=3,
    cols=3,
    specs=[
        [{}, {}, {"type": "domain"}],
        [{}, {}, {}],
        [{"colspan": 3}, None, None],
    ],
    column_widths=[0.36, 0.34, 0.30],
    row_heights=[0.30, 0.31, 0.39],
    horizontal_spacing=0.085,
    vertical_spacing=0.13,
    subplot_titles=(
        "6 wilayah dengan kemiskinan tertinggi",
        "Sebaran angka kemiskinan",
        f"Perbandingan median · {median_poverty:.2f}%",
        "Distribusi kumulatif wilayah",
        "Pengeluaran dan kemiskinan",
        "Rentang kemiskinan · 5 provinsi",
        "Korelasi Spearman antarindikator",
    ),
)
overview.add_trace(
    go.Bar(
        x=top["Kemiskinan"],
        y=top["KabKota"],
        orientation="h",
        marker={"color": top["Kemiskinan"], "colorscale": [[0, "#f1d27b"], [1, "#d9573f"]]},
        customdata=top["Provinsi"],
        hovertemplate="%{y}<br>%{customdata}<br><b>%{x:.2f}%</b><extra></extra>",
    ),
    row=1,
    col=1,
)
overview.add_trace(
    go.Histogram(
        x=bersih["Kemiskinan"],
        nbinsx=20,
        marker={"color": "#55765c", "line": {"color": "#fff8e8", "width": 1}},
        hovertemplate="%{x:.1f}%<br>%{y} wilayah<extra></extra>",
    ),
    row=1,
    col=2,
)
overview.add_trace(
    go.Pie(
        labels=["≤ median", "> median"],
        values=[
            int((bersih["Kemiskinan"] <= median_poverty).sum()),
            int((bersih["Kemiskinan"] > median_poverty).sum()),
        ],
        hole=0.63,
        sort=False,
        marker={"colors": ["#55765c", "#d9573f"], "line": {"color": "#fff8e8", "width": 3}},
        textinfo="label+percent",
        textposition="outside",
        textfont={"size": 8},
        hovertemplate="%{label}<br>%{value} wilayah (%{percent})<extra></extra>",
    ),
    row=1,
    col=3,
)
overview.add_trace(
    go.Scatter(
        x=sorted_poverty,
        y=np.arange(1, len(sorted_poverty) + 1) / len(sorted_poverty) * 100,
        mode="lines",
        fill="tozeroy",
        line={"color": "#55765c", "width": 2.5},
        fillcolor="rgba(85,118,92,0.2)",
        hovertemplate="Kemiskinan %{x:.2f}%<br>Wilayah kumulatif %{y:.1f}%<extra></extra>",
    ),
    row=2,
    col=1,
)
overview.add_trace(
    go.Scatter(
        x=bersih["Pengeluaran"],
        y=bersih["Kemiskinan"],
        mode="markers",
        marker={"size": 6, "color": "#d9573f", "opacity": 0.68, "line": {"width": 0.5, "color": "#fff8e8"}},
        customdata=bersih[["KabKota", "Provinsi"]],
        hovertemplate="%{customdata[0]}, %{customdata[1]}<br>Pengeluaran %{x:,.0f}<br>Kemiskinan %{y:.2f}%<extra></extra>",
    ),
    row=2,
    col=2,
)
for index, province in enumerate(province_order):
    province_data = bersih.loc[bersih["Provinsi"].eq(province), "Kemiskinan"]
    overview.add_trace(
        go.Box(
            x=province_data,
            name=province,
            orientation="h",
            boxpoints="outliers",
            marker={"color": ["#d9573f", "#55765c", "#e0ae4f", "#647da2", "#ac6871"][index], "size": 4},
            line={"color": ["#d9573f", "#55765c", "#e0ae4f", "#647da2", "#ac6871"][index], "width": 1.5},
            hovertemplate="%{x:.2f}%<extra>%{fullData.name}</extra>",
        ),
        row=2,
        col=3,
    )
overview.add_trace(
    go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        zmin=-1,
        zmax=1,
        zmid=0,
        colorscale=[[0, "#c95748"], [0.5, "#f4e8cc"], [1, "#55765c"]],
        text=corr.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 9, "color": "#342820"},
        colorbar={
            "title": {"text": "ρ", "font": {"size": 10}},
            "len": 0.28,
            "y": 0.14,
            "yanchor": "middle",
            "x": 1.01,
        },
        hovertemplate="%{y} × %{x}<br>Spearman ρ = %{z:.2f}<extra></extra>",
    ),
    row=3,
    col=1,
)
overview.update_layout(
    height=940,
    title={
        "text": "RUANG DATA 01  /  POTRET KEMISKINAN KABUPATEN/KOTA",
        "x": 0.025,
        "xanchor": "left",
        "font": {"family": "Fraunces, Georgia, serif", "size": 23, "color": "#342820"},
    },
    font={"family": "'Space Mono', Consolas, monospace", "size": 10, "color": "#342820"},
    paper_bgcolor="#fff8e8",
    plot_bgcolor="#fff8e8",
    margin={"l": 36, "r": 28, "t": 88, "b": 34},
    showlegend=False,
    hoverlabel={"bgcolor": "#fff8e8", "font": {"color": "#342820", "family": "Space Mono"}},
)
overview.update_annotations(font={"family": "Space Mono, monospace", "size": 10, "color": "#55765c"})
overview.update_xaxes(showgrid=True, gridcolor="#e8dcc5", zeroline=False, tickfont={"size": 8})
overview.update_yaxes(showgrid=False, tickfont={"size": 8})
overview.update_xaxes(title_text="Kemiskinan (%)", row=1, col=1)
overview.update_yaxes(categoryorder="total ascending", row=1, col=1)
overview.update_xaxes(title_text="Kemiskinan (%)", row=1, col=2)
overview.update_yaxes(title_text="Jumlah", row=1, col=2)
overview.update_xaxes(title_text="Kemiskinan (%)", row=2, col=1)
overview.update_yaxes(title_text="Wilayah kumulatif (%)", row=2, col=1)
overview.update_xaxes(title_text="Pengeluaran", row=2, col=2)
overview.update_yaxes(title_text="Kemiskinan (%)", row=2, col=2)
overview.update_xaxes(title_text="Kemiskinan (%)", row=2, col=3)
overview.update_xaxes(side="bottom", row=3, col=1, tickfont={"size": 8})
overview.update_yaxes(autorange="reversed", row=3, col=1, tickfont={"size": 8})
overview_div = overview.to_html(
    full_html=False,
    include_plotlyjs="cdn",
    div_id="overview-dashboard",
    config={
        "responsive": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "ruang-data-kemiskinan",
            "height": 1500,
            "width": 1800,
            "scale": 2,
        },
    },
)

dashboard_html = f"""<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Dashboard interaktif analisis kemiskinan kabupaten/kota di Indonesia.">
  <title>Ruang Data 01 | Kemiskinan Kabupaten/Kota</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,800&family=Space+Mono:wght@400;700&display=swap');
    :root {{ color-scheme: light; --ink:#342820; --muted:#765e4a; --line:#4c3b2e; --paper:#f4e8cc; --cream:#fff8e8; --tomato:#d9573f; --olive:#55765c; --gold:#edbd55; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background-color:#f1e4c9; background-image:radial-gradient(rgba(94,71,48,.11) .8px,transparent .8px); background-size:12px 12px; color:var(--ink); font-family:"Space Mono",Consolas,monospace; }}
    .wrap {{ max-width:1420px; margin:auto; padding:32px 28px 56px; }}
    .hero {{ position:relative; overflow:hidden; padding:44px 48px; border:3px solid var(--ink);
      border-radius:7px 38px 7px 38px; color:#fff8e8; background:linear-gradient(112deg,#3b5d4c 0%,#55765c 64%,#8c8a4d 100%);
      box-shadow:10px 10px 0 var(--tomato); }}
    .hero:after {{ content:""; position:absolute; width:270px; height:270px; right:-80px; top:-128px;
      border:2px solid rgba(255,248,232,.55); border-radius:50%;
      box-shadow:0 0 0 18px rgba(255,248,232,.12),0 0 0 38px rgba(255,248,232,.09),0 0 0 58px rgba(255,248,232,.06); }}
    .eyebrow {{ color:var(--tomato); font-size:10px; font-weight:700; letter-spacing:.12em; margin:0 0 8px; font-family:"Space Mono",monospace; }}
    .hero .eyebrow {{ color:#f6d878; }}
    h1 {{ font-family:"Fraunces",Georgia,serif; font-size:clamp(32px,4vw,52px); line-height:1.04; letter-spacing:-.035em; max-width:800px; margin:14px 0; }}
    .hero p {{ color:#f8efda; line-height:1.75; max-width:760px; margin:0; font-size:14px; }}
    .source-pill {{ display:inline-flex; margin-top:22px; border:2px solid #f6d878; border-radius:4px 18px 4px 18px; padding:9px 14px; font-size:11px; color:#fff8e8; background:rgba(38,54,43,.32); }}
    .stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:15px; margin:24px 0; }}
    .stat {{ background:var(--cream); border:2px solid var(--ink); border-radius:16px 4px 16px 4px; padding:19px 21px; box-shadow:5px 5px 0 rgba(52,40,32,.18); }}
    .stat:nth-child(2) {{ background:#f6d9c4; }}
    .stat:nth-child(3) {{ background:#e0e8ce; }}
    .stat:nth-child(4) {{ background:#f3d880; }}
    .stat-label {{ color:#705743; font-size:10px; font-family:"Space Mono",monospace; text-transform:uppercase; letter-spacing:.05em; }}
    .stat-value {{ display:block; font-family:"Fraunces",Georgia,serif; font-size:32px; font-weight:800; letter-spacing:-.035em; margin-top:9px; color:var(--ink); }}
    .stat-note {{ color:#765e4a; font-size:10px; margin-top:5px; }}
    .insights {{ display:grid; grid-template-columns:1.1fr 1fr 1fr; gap:14px; margin:16px 0 28px; }}
    .insight {{ padding:17px 19px; border-radius:4px 18px 4px 18px; background:#dce7d1; border:2px solid var(--ink); box-shadow:4px 4px 0 rgba(52,40,32,.15); }}
    .insight:nth-child(2) {{ background:#f4d2bd; }}
    .insight:nth-child(3) {{ background:#f1d77d; }}
    .insight strong {{ display:block; font-family:"Fraunces",Georgia,serif; font-size:16px; margin-bottom:7px; color:var(--ink); }}
    .insight p {{ color:#604c3e; line-height:1.6; font-size:11px; margin:0; }}
    .section-title {{ display:flex; align-items:end; justify-content:space-between; gap:20px; margin:38px 3px 15px; }}
    .section-title h2 {{ margin:0; font-family:"Fraunces",Georgia,serif; font-size:25px; letter-spacing:-.025em; color:var(--ink); }}
    .section-title p {{ color:var(--muted); margin:5px 0 0; font-size:10px; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:17px; }}
    .panel {{ min-width:0; padding:20px 20px 12px; border-radius:18px 5px 18px 5px; background:var(--cream);
      border:2px solid var(--ink); box-shadow:5px 5px 0 rgba(52,40,32,.16); }}
    .panel-heading h2 {{ font-family:"Fraunces",Georgia,serif; color:var(--ink); font-size:19px; margin:0 0 6px; letter-spacing:-.02em; }}
    .panel-heading .muted {{ margin:0; color:var(--muted); font-size:10px; line-height:1.55; }}
    .panel-heading .eyebrow {{ margin-bottom:7px; }}
    .note {{ margin-top:22px; padding:18px 20px; border:2px solid var(--ink); border-left:9px solid var(--tomato); background:#f5d6bd;
      color:#49372c; border-radius:4px 18px 4px 18px; font-size:11px; line-height:1.7; box-shadow:5px 5px 0 rgba(52,40,32,.14); }}
    footer {{ color:#705743; font-size:10px; padding:25px 4px 0; }}
    @media(max-width:900px) {{ .stats {{grid-template-columns:repeat(2,1fr)}} .insights {{grid-template-columns:1fr}} .grid {{grid-template-columns:1fr}} .hero {{padding:32px 28px}} }}
    @media(max-width:520px) {{ .wrap {{padding:16px 12px 36px}} .stats {{gap:9px}} .stat {{padding:15px}} .stat-value {{font-size:23px}} .panel {{padding:15px 12px 8px}} }}
  </style>
</head>
<body>
  <main class="wrap">
    <header class="hero">
      <p class="eyebrow">RUANG DATA · ANALISIS SOSIAL · VOL. 01</p>
      <h1>Potret kemiskinan kabupaten/kota.</h1>
      <p>Potret indikator kabupaten/kota untuk melihat sebaran kemiskinan dan hubungannya dengan kondisi sosial-ekonomi. Gunakan visualisasi sebagai titik awal eksplorasi, bukan kesimpulan sebab-akibat.</p>
      <span class="source-pill">Dataset kabupaten/kota · {len(bersih)} wilayah · Korelasi Spearman</span>
    </header>

    <section class="stats" aria-label="Ringkasan indikator">
      <article class="stat"><span class="stat-label">Wilayah dianalisis</span><strong class="stat-value">{len(bersih):,}</strong><div class="stat-note">setelah deduplikasi</div></article>
      <article class="stat"><span class="stat-label">Rata-rata kemiskinan</span><strong class="stat-value">{mean_poverty:.2f}%</strong><div class="stat-note">rata-rata kabupaten/kota</div></article>
      <article class="stat"><span class="stat-label">Rata-rata IPM</span><strong class="stat-value">{bersih["IPM"].mean():.1f}</strong><div class="stat-note">indeks pembangunan manusia</div></article>
      <article class="stat"><span class="stat-label">Nilai tertinggi</span><strong class="stat-value">{top_location["Kemiskinan"]:.2f}%</strong><div class="stat-note">{top_location["KabKota"]}, {top_location["Provinsi"]}</div></article>
    </section>

    <div class="section-title"><div><p class="eyebrow">INTISARI</p><h2>Apa yang tampak dari data?</h2></div></div>
    <section class="insights">
      <article class="insight"><strong>Wilayah tertinggi: {top_location["KabKota"]}</strong><p>Angka kemiskinan tercatat {top_location["Kemiskinan"]:.2f}% setelah proses validasi dan imputasi.</p></article>
      <article class="insight"><strong>Hubungan negatif terkuat: {negative_feature}</strong><p>Spearman ρ = {corr.loc[negative_feature, "Kemiskinan"]:.2f} terhadap kemiskinan. Ini asosiasi, bukan bukti sebab-akibat.</p></article>
      <article class="insight"><strong>Hubungan positif terkuat: {positive_feature}</strong><p>Spearman ρ = {corr.loc[positive_feature, "Kemiskinan"]:.2f} terhadap kemiskinan pada data yang telah dibersihkan.</p></article>
    </section>

    <div class="section-title"><div><p class="eyebrow">SATU PAPAN · TUJUH PERSPEKTIF</p><h2>Potret data dalam satu pandangan</h2><p>Arahkan kursor untuk detail. Gunakan tombol unduh grafik untuk menyimpan PNG berkualitas tinggi.</p></div></div>
    <section class="panel overview-panel">
      {overview_div}
    </section>

    <aside class="note"><strong>Catatan kualitas data.</strong> 16 baris duplikat persis dihapus. Setelah deduplikasi, {rows_with_anomaly} baris memiliki setidaknya satu nilai di luar batas validasi dan nilai tersebut diimputasi dengan median provinsi. Karena itu, angka pada dashboard perlu dibaca bersama keterbatasan pembersihan dan diverifikasi ke sumber resmi sebelum digunakan untuk keputusan.</aside>
    <footer>Analisis deskriptif · Korelasi bukan sebab-akibat · Dataset dan metode dijelaskan di README proyek.</footer>
  </main>
</body>
</html>"""
(ROOT / "dashboard/dashboard_kemiskinan.html").write_text(
    dashboard_html,
    encoding="utf-8",
)

app = '''from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Ruang Data 01 | Kemiskinan",
    page_icon="📻",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR.parent / "data/clean/data_bersih.csv")
features = [
    "Kemiskinan", "RLS", "Pengeluaran", "IPM", "UHH",
    "Sanitasi", "AirMinum", "TPT", "TPAK", "PDRB",
]

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,800&family=Space+Mono:wght@400;700&display=swap');
  :root { color-scheme:light; }
  html, body, [class*="css"] { font-family:"Space Mono",Consolas,monospace; }
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color:#f1e4c9 !important;
    background-image:radial-gradient(rgba(94,71,48,.09) .8px,transparent .8px) !important;
    background-size:12px 12px !important; color:#342820 !important;
  }
  header[data-testid="stHeader"] { background:rgba(241,228,201,.95) !important; }
  header[data-testid="stHeader"] * { color:#342820 !important; }
  .block-container { max-width:1480px; padding:1.7rem 2.2rem 3rem; }
  .stApp p, .stApp label, .stApp li, .stApp [data-testid="stCaptionContainer"] {
    color:#705743 !important;
  }
  .stApp h1, .stApp h2, .stApp h3, .stApp h4,
  .stApp [data-testid="stMarkdownContainer"] h1,
  .stApp [data-testid="stMarkdownContainer"] h2,
  .stApp [data-testid="stMarkdownContainer"] h3 {
    color:#342820 !important; font-family:"Fraunces",Georgia,serif;
  }
  section[data-testid="stMain"] .hero h1 { color:#fff8e8 !important; }
  section[data-testid="stMain"] .hero p { color:#f8efda !important; }
  .hero { position:relative; overflow:hidden; padding:2.5rem 2.7rem; border:3px solid #342820;
    border-radius:7px 38px 7px 38px; color:#fff8e8;
    background:linear-gradient(112deg,#3b5d4c 0%,#55765c 64%,#8c8a4d 100%);
    margin-bottom:1.1rem; box-shadow:10px 10px 0 #d9573f; }
  .hero:after { content:""; position:absolute; width:270px; height:270px; right:-80px; top:-128px;
    border:2px solid rgba(255,248,232,.55); border-radius:50%;
    box-shadow:0 0 0 18px rgba(255,248,232,.12),0 0 0 38px rgba(255,248,232,.09),0 0 0 58px rgba(255,248,232,.06); }
  .hero-kicker { display:inline-block; color:#f6d878; font-size:.67rem; font-weight:700;
    letter-spacing:.12em; border:2px solid rgba(246,216,120,.72); border-radius:4px 14px 4px 14px;
    padding:.42rem .72rem; font-family:"Space Mono",monospace; }
  .hero h1 { position:relative; z-index:1; font-family:"Fraunces",Georgia,serif;
    font-size:clamp(2.1rem,4vw,3.25rem); line-height:1.04; letter-spacing:-.035em;
    max-width:880px; margin:1rem 0 .7rem; }
  .hero p { position:relative; z-index:1; max-width:830px; margin-bottom:0; line-height:1.75; font-size:.9rem; }
  .stApp [data-testid="stSidebar"] .side-brand span { display:block; color:#f6d878 !important;
    font-size:.68rem; font-weight:700; letter-spacing:.12em; margin-bottom:.38rem; }
  .stApp [data-testid="stSidebar"] .side-brand strong { display:block; color:#fff8e8 !important;
    font-family:"Fraunces",Georgia,serif; font-size:1.1rem; line-height:1.35; }
  section[data-testid="stSidebar"] { background:#344d3f !important; border-right:3px solid #342820; }
  section[data-testid="stSidebar"] * { color:#f8efda !important; }
  section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color:#fff8e8 !important; }
  div[data-testid="stMultiSelect"] label, div[data-testid="stMultiSelect"] label p {
    color:#49372c !important; font-weight:700; }
  div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background:#fff8e8 !important; border:2px solid #4c3b2e !important; border-radius:4px 14px 4px 14px !important;
  }
  div[data-testid="stMultiSelect"] [data-baseweb="select"] * { color:#342820 !important; }
  [data-baseweb="popover"] *, [role="listbox"] * { color:#342820 !important; }
  .scope-card { margin:.6rem 0 1rem; padding:.78rem 1rem; min-height:68px;
    background:#f1d77d; border:2px solid #342820; border-radius:4px 17px 4px 17px;
    box-shadow:4px 4px 0 rgba(52,40,32,.16); }
  .scope-card span { display:block; color:#705743; font-size:.62rem; font-weight:700; letter-spacing:.1em; }
  .scope-card strong { display:inline-block; color:#342820; font-size:.88rem; margin:.12rem .45rem .05rem 0; }
  .scope-card small { color:#705743; font-size:.72rem; }
  div[data-testid="stMetric"] { min-height:122px; background:#fff8e8 !important;
    border:2px solid #342820; border-top:5px solid #d9573f; padding:1rem 1.15rem;
    border-radius:16px 4px 16px 4px; box-shadow:5px 5px 0 rgba(52,40,32,.17); }
  div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p {
    color:#705743 !important; font-family:"Space Mono",monospace; font-size:.75rem; font-weight:700; }
  div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
    color:#342820 !important; font-family:"Fraunces",Georgia,serif; }
  div[data-testid="stMetricDelta"], div[data-testid="stMetricDelta"] * { color:#55765c !important; }
  div[data-testid="stTabs"] button[data-baseweb="tab"] { color:#705743 !important;
    border-bottom:2px solid transparent !important; font-family:"Space Mono",monospace; }
  div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color:#a84232 !important; border-bottom:3px solid #d9573f !important; }
  div[data-testid="stTabs"] button[data-baseweb="tab"] p { color:inherit !important; }
  div[data-testid="stAlert"] { border:2px solid #4c3b2e; border-radius:4px 16px 4px 16px; }
  div[data-testid="stAlert"] p { color:#49372c !important; }
  .insight { position:relative; overflow:hidden; padding:1.1rem 1.2rem; border-radius:4px 17px 4px 17px;
    background:#dce7d1; border:2px solid #342820; border-left:7px solid #55765c; min-height:112px;
    box-shadow:4px 4px 0 rgba(52,40,32,.15); }
  .insight b { color:#342820 !important; display:block; font-family:"Fraunces",Georgia,serif;
    font-size:.98rem; margin-bottom:.42rem; }
  .insight span { color:#604c3e !important; font-size:.8rem; line-height:1.55; }
  [data-testid="stDataFrame"] { border:2px solid #342820; border-radius:4px 17px 4px 17px; }
  @media(max-width:700px) {
    .block-container { padding:1rem 1rem 2rem; }
    .hero { padding:1.65rem 1.35rem; border-radius:5px 24px 5px 24px; }
    div[data-testid="stMetric"] { min-height:105px; padding:.75rem; }
  }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<section class="hero">
  <div class="hero-kicker">RUANG DATA · ANALISIS SOSIAL · VOL. 01</div>
  <h1>Potret kemiskinan<br>kabupaten/kota.</h1>
  <p>Jelajahi sebaran kemiskinan dan indikator sosial-ekonomi kabupaten/kota. Gunakan filter provinsi untuk membandingkan wilayah. Korelasi adalah asosiasi, bukan sebab-akibat.</p>
</section>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        '<div class="side-brand"><span>RUANG DATA</span>'
        '<strong>Kemiskinan kabupaten/kota</strong></div>',
        unsafe_allow_html=True,
    )
    st.caption("DASHBOARD DATA SOSIAL")
    st.divider()
    st.caption("Metode korelasi: Spearman")
    st.caption("Nilai di luar batas validasi telah diimputasi dengan median provinsi.")

filter_col, scope_col = st.columns([1.7, 1])
with filter_col:
    selected_provinces = st.multiselect(
        "Saring berdasarkan provinsi",
        sorted(df["Provinsi"].unique()),
        help="Pilih satu atau beberapa provinsi. Hapus semua pilihan untuk kembali ke seluruh wilayah.",
        placeholder="Semua provinsi",
    )
with scope_col:
    st.markdown(
        f'<div class="scope-card"><span>CAKUPAN SAAT INI</span><strong>'
        f'{"Semua provinsi" if not selected_provinces else f"{len(selected_provinces)} provinsi dipilih"}</strong>'
        f'<small>{len(df[df["Provinsi"].isin(selected_provinces)]) if selected_provinces else len(df):,} kabupaten/kota</small></div>',
        unsafe_allow_html=True,
    )

d = df[df["Provinsi"].isin(selected_provinces)] if selected_provinces else df
correlation = d[features].corr(method="spearman")
province_means = d.groupby("Provinsi")["Kemiskinan"].mean().sort_values(ascending=False)
top_location = d.loc[d["Kemiskinan"].idxmax()]
strongest = correlation["Kemiskinan"].drop("Kemiskinan").dropna()
negative_feature = strongest.idxmin() if not strongest.empty else None

k1, k2 = st.columns(2)
k1.metric("Wilayah dianalisis", f"{len(d):,}")
k2.metric("Rata-rata kemiskinan", f"{d['Kemiskinan'].mean():.2f}%")
k3, k4 = st.columns(2)
k3.metric("Rata-rata IPM", f"{d['IPM'].mean():.1f}")
k4.metric("Tertinggi di pilihan", f"{top_location['Kemiskinan']:.2f}%", top_location["KabKota"])

st.markdown("### Intisari wilayah terpilih")
i1, i2, i3 = st.columns(3)
with i1:
    st.markdown(
        f'<div class="insight"><b>Wilayah tertinggi</b><span>{top_location["KabKota"]}, '
        f'{top_location["Provinsi"]}: {top_location["Kemiskinan"]:.2f}%.</span></div>',
        unsafe_allow_html=True,
    )
with i2:
    st.markdown(
        f'<div class="insight"><b>Rata-rata provinsi tertinggi</b><span>{province_means.index[0]}: '
        f'{province_means.iloc[0]:.2f}%.</span></div>',
        unsafe_allow_html=True,
    )
with i3:
    if negative_feature:
        st.markdown(
            f'<div class="insight"><b>Asosiasi negatif terkuat</b><span>{negative_feature} dan kemiskinan '
            f'(Spearman ρ = {strongest[negative_feature]:.2f}); bukan hubungan sebab-akibat.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="insight"><b>Asosiasi</b><span>Korelasi belum dapat dihitung untuk pilihan wilayah ini.</span></div>',
            unsafe_allow_html=True,
        )

summary_tab, correlation_tab, data_tab = st.tabs(
    ["Ringkasan", "Korelasi & hubungan", "Data wilayah"]
)

with summary_tab:
    st.markdown("### Satu papan, tujuh sudut pandang")
    st.caption(
        "Ringkasan ini menggabungkan peringkat, sebaran, median, rentang provinsi, hubungan indikator, "
        "dan korelasi. Arahkan kursor ke setiap panel untuk membaca nilai rinci."
    )
    median = d["Kemiskinan"].median()
    ranked = d.nlargest(6, "Kemiskinan").sort_values("Kemiskinan")
    sorted_poverty = d["Kemiskinan"].sort_values().reset_index(drop=True)
    top_province_order = (
        d.groupby("Provinsi")["Kemiskinan"]
        .median()
        .sort_values()
        .tail(5)
        .index
        .tolist()
    )

    overview = make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{}, {}, {"type": "domain"}],
            [{}, {}, {}],
            [{"colspan": 3}, None, None],
        ],
        column_widths=[0.36, 0.34, 0.30],
        row_heights=[0.30, 0.31, 0.39],
        horizontal_spacing=0.085,
        vertical_spacing=0.13,
        subplot_titles=(
            "6 wilayah tertinggi",
            "Sebaran angka",
            f"Posisi terhadap median · {median:.2f}%",
            "Distribusi kumulatif",
            "Pengeluaran × kemiskinan",
            "Rentang · 5 provinsi",
            "Korelasi Spearman antarindikator",
        ),
    )
    overview.add_trace(
        go.Bar(
            x=ranked["Kemiskinan"],
            y=ranked["KabKota"],
            orientation="h",
            marker={"color": ranked["Kemiskinan"], "colorscale": [[0, "#f1d27b"], [1, "#d9573f"]]},
            customdata=ranked["Provinsi"],
            hovertemplate="%{y}<br>%{customdata}<br><b>%{x:.2f}%</b><extra></extra>",
        ),
        row=1,
        col=1,
    )
    overview.add_trace(
        go.Histogram(
            x=d["Kemiskinan"],
            nbinsx=20,
            marker={"color": "#55765c", "line": {"color": "#fff8e8", "width": 1}},
            hovertemplate="%{x:.1f}%<br>%{y} wilayah<extra></extra>",
        ),
        row=1,
        col=2,
    )
    overview.add_trace(
        go.Pie(
            labels=["≤ median", "> median"],
            values=[
                int((d["Kemiskinan"] <= median).sum()),
                int((d["Kemiskinan"] > median).sum()),
            ],
            hole=0.63,
            sort=False,
            marker={"colors": ["#55765c", "#d9573f"], "line": {"color": "#fff8e8", "width": 3}},
            textinfo="label+percent",
            textposition="outside",
            textfont={"size": 8},
            hovertemplate="%{label}<br>%{value} wilayah (%{percent})<extra></extra>",
        ),
        row=1,
        col=3,
    )
    overview.add_trace(
        go.Scatter(
            x=sorted_poverty,
            y=np.arange(1, len(sorted_poverty) + 1) / len(sorted_poverty) * 100,
            mode="lines",
            fill="tozeroy",
            line={"color": "#55765c", "width": 2.5},
            fillcolor="rgba(85,118,92,0.2)",
            hovertemplate="Kemiskinan %{x:.2f}%<br>Wilayah kumulatif %{y:.1f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    overview.add_trace(
        go.Scatter(
            x=d["Pengeluaran"],
            y=d["Kemiskinan"],
            mode="markers",
            marker={"size": 6, "color": "#d9573f", "opacity": 0.68, "line": {"width": 0.5, "color": "#fff8e8"}},
            customdata=d[["KabKota", "Provinsi"]],
            hovertemplate="%{customdata[0]}, %{customdata[1]}<br>Pengeluaran %{x:,.0f}<br>Kemiskinan %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=2,
    )
    box_colors = ["#d9573f", "#55765c", "#e0ae4f", "#647da2", "#ac6871"]
    for index, province in enumerate(top_province_order):
        province_data = d.loc[d["Provinsi"].eq(province), "Kemiskinan"]
        overview.add_trace(
            go.Box(
                x=province_data,
                name=province,
                orientation="h",
                boxpoints="outliers",
                marker={"color": box_colors[index], "size": 4},
                line={"color": box_colors[index], "width": 1.5},
                hovertemplate="%{x:.2f}%<extra>%{fullData.name}</extra>",
            ),
            row=2,
            col=3,
        )
    overview.add_trace(
        go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.index,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[[0, "#c95748"], [0.5, "#f4e8cc"], [1, "#55765c"]],
            text=correlation.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 9, "color": "#342820"},
            colorbar={
                "title": {"text": "ρ", "font": {"size": 10}},
                "len": 0.28,
                "y": 0.14,
                "yanchor": "middle",
                "x": 1.01,
            },
            hovertemplate="%{y} × %{x}<br>Spearman ρ = %{z:.2f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    overview.update_layout(
        height=940,
        title={
            "text": "RUANG DATA 01  /  POTRET KEMISKINAN KABUPATEN/KOTA",
            "x": 0.025,
            "xanchor": "left",
            "font": {"family": "Fraunces, Georgia, serif", "size": 23, "color": "#342820"},
        },
        font={"family": "'Space Mono', Consolas, monospace", "size": 10, "color": "#342820"},
        paper_bgcolor="#fff8e8",
        plot_bgcolor="#fff8e8",
        margin={"l": 36, "r": 28, "t": 88, "b": 34},
        showlegend=False,
        hoverlabel={"bgcolor": "#fff8e8", "font": {"color": "#342820", "family": "Space Mono"}},
    )
    overview.update_annotations(font={"family": "Space Mono, monospace", "size": 10, "color": "#55765c"})
    overview.update_xaxes(showgrid=True, gridcolor="#e8dcc5", zeroline=False, tickfont={"size": 8})
    overview.update_yaxes(showgrid=False, tickfont={"size": 8})
    overview.update_xaxes(title_text="Kemiskinan (%)", row=1, col=1)
    overview.update_yaxes(categoryorder="total ascending", row=1, col=1)
    overview.update_xaxes(title_text="Kemiskinan (%)", row=1, col=2)
    overview.update_yaxes(title_text="Jumlah", row=1, col=2)
    overview.update_xaxes(title_text="Kemiskinan (%)", row=2, col=1)
    overview.update_yaxes(title_text="Wilayah kumulatif (%)", row=2, col=1)
    overview.update_xaxes(title_text="Pengeluaran", row=2, col=2)
    overview.update_yaxes(title_text="Kemiskinan (%)", row=2, col=2)
    overview.update_xaxes(title_text="Kemiskinan (%)", row=2, col=3)
    overview.update_xaxes(side="bottom", row=3, col=1, tickfont={"size": 8})
    overview.update_yaxes(autorange="reversed", row=3, col=1, tickfont={"size": 8})

    st.plotly_chart(
        overview,
        width="stretch",
        theme=None,
        config={
            "responsive": True,
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "ruang-data-kemiskinan",
                "height": 1500,
                "width": 1800,
                "scale": 2,
            },
        },
    )
    st.caption(
        "Kurva area adalah distribusi kumulatif lintas wilayah, bukan tren waktu. "
        "Donut membagi wilayah terhadap median pilihan; kategori ini bukan klasifikasi resmi."
    )
    st.info(
        "Gunakan filter provinsi di atas untuk memperbarui semua panel. "
        "Klik ikon kamera pada grafik untuk mengunduh satu gambar PNG komposit beresolusi tinggi."
    )

with correlation_tab:
    st.markdown("### Peta korelasi Spearman")
    st.caption(
        "Rentang -1 hingga +1. Nilai negatif berarti kedua variabel cenderung bergerak berlawanan; "
        "nilai positif cenderung bergerak searah."
    )
    heatmap = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=[[0, "#c95748"], [0.5, "#f4e8cc"], [1, "#55765c"]],
        zmin=-1,
        zmax=1,
        color_continuous_midpoint=0,
        title="Hubungan antarindikator dalam wilayah terpilih",
    )
    heatmap.update_layout(
        template="plotly_white",
        paper_bgcolor="#fff8e8",
        plot_bgcolor="#fff8e8",
        font_color="#342820",
        title_font_color="#342820",
        height=720,
        margin=dict(l=8, r=8, t=65, b=8),
        coloraxis_colorbar_title="Spearman ρ",
    )
    st.plotly_chart(heatmap, width="stretch", theme=None)

    if not strongest.empty:
        assoc = strongest.reindex(strongest.abs().sort_values(ascending=False).index)
        assoc_fig = px.bar(
            x=assoc.values,
            y=assoc.index,
            orientation="h",
            color=assoc.values,
            color_continuous_scale=[[0, "#c95748"], [0.5, "#f4e8cc"], [1, "#55765c"]],
            range_color=[-1, 1],
            title="Korelasi setiap indikator dengan kemiskinan",
            labels={"x": "Spearman ρ", "y": "Indikator", "color": "ρ"},
        )
        assoc_fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#fff8e8",
            plot_bgcolor="#fff8e8",
            font_color="#342820",
            title_font_color="#342820",
            coloraxis_showscale=False,
            xaxis=dict(range=[-1, 1], zeroline=True, zerolinecolor="#334155"),
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=8, r=8, t=55, b=8),
        )
        st.plotly_chart(assoc_fig, width="stretch", theme=None)
    else:
        st.info("Tidak cukup variasi data untuk menghitung korelasi terhadap kemiskinan pada filter ini.")

    scatter = px.scatter(
        d,
        x="Pengeluaran",
        y="Kemiskinan",
        color="Provinsi",
        hover_name="KabKota",
        hover_data={"Pengeluaran": ":,.0f", "Kemiskinan": ":.2f"},
        color_discrete_sequence=["#d9573f", "#55765c", "#e0ae4f", "#647da2", "#ac6871", "#87714b", "#39848a", "#a45b37"],
        title="Pengeluaran dan kemiskinan",
    )
    scatter.update_layout(
        template="plotly_white",
        paper_bgcolor="#fff8e8",
        plot_bgcolor="#fff8e8",
        font_color="#342820",
        title_font_color="#342820",
        xaxis_title="Pengeluaran per kapita",
        yaxis_title="Penduduk miskin (%)",
        margin=dict(l=8, r=8, t=55, b=8),
    )
    st.plotly_chart(scatter, width="stretch", theme=None)
    st.warning(
        "Korelasi bersifat deskriptif. Pola dapat dipengaruhi kualitas data, "
        "variabel lain, dan imputasi nilai; bukan bukti sebab-akibat."
    )

with data_tab:
    st.markdown("### Jelajahi tabel data")
    st.caption(f"Menampilkan {len(d):,} wilayah berdasarkan filter yang dipilih.")
    st.dataframe(
        d.sort_values("Kemiskinan", ascending=False),
        width="stretch",
        hide_index=True,
    )

st.divider()
st.caption(
    "Catatan kualitas: 16 duplikat persis dihapus; 61 baris setelah deduplikasi "
    "memiliki nilai di luar batas validasi dan diimputasi dengan median provinsi. "
    "Verifikasi terhadap sumber resmi sebelum digunakan untuk keputusan."
)
'''
(ROOT / "dashboard/app.py").write_text(app, encoding="utf-8")
(ROOT / "dashboard/requirements.txt").write_text(
    "streamlit\npandas\nplotly\n",
    encoding="utf-8",
)
(ROOT / "dashboard/.streamlit/config.toml").write_text(
    '[theme]\n'
    'base = "light"\n'
    'primaryColor = "#d9573f"\n'
    'backgroundColor = "#f1e4c9"\n'
    'secondaryBackgroundColor = "#fff8e8"\n'
    'textColor = "#342820"\n'
    'font = "sans serif"\n',
    encoding="utf-8",
)

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Analisis Kemiskinan Kabupaten/Kota\n\n"
        "Notebook ini memuat proses kualitas data, pembersihan, EDA, korelasi, "
        "model klasifikasi eksploratif, dan ringkasan temuan."
    ),
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import plotly.express as px\n"
        "import plotly.graph_objects as go\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.metrics import classification_report\n\n"
        "df = pd.read_csv('../data/raw/Dataset_Kemiskinan.csv', sep=';', decimal=',')\n"
        "df.shape"
    ),
    nbf.v4.new_markdown_cell(
        "## 1. Pemeriksaan kualitas data\n"
        "Data diperiksa untuk duplikat, nilai kosong, dan nilai di luar rentang wajar."
    ),
    nbf.v4.new_code_cell(
        "print('Baris kembar persis:', df.duplicated().sum())\n"
        "print('Nilai kosong:')\n"
        "print(df.isna().sum())\n"
        "df.describe(include='all').T"
    ),
    nbf.v4.new_markdown_cell("## 2. Pembersihan data"),
    nbf.v4.new_code_cell(
        "batas = {\n"
        "    'Kemiskinan': (0, 60), 'RLS': (3, 13),\n"
        "    'Pengeluaran': (4000, 30000), 'IPM': (45, 90),\n"
        "    'UHH': (55, 80), 'Sanitasi': (5, 100),\n"
        "    'AirMinum': (5, 100), 'TPT': (0, 15),\n"
        "    'TPAK': (50, 100), 'PDRB': (1, np.inf)\n"
        "}\n"
        "df = df.drop_duplicates().reset_index(drop=True)\n"
        "anomali = pd.DataFrame({c: df[c].notna() & ~df[c].between(*r)\n"
        "                         for c, r in batas.items()})\n"
        "print(anomali.sum().sort_values(ascending=False))\n"
        "print('Baris dengan minimal satu anomali:', anomali.any(axis=1).sum())\n\n"
        "bersih = df.copy()\n"
        "for c, r in batas.items():\n"
        "    bersih.loc[~bersih[c].between(*r), c] = np.nan\n"
        "for c in batas:\n"
        "    bersih[c] = bersih[c].fillna(bersih.groupby('Provinsi')[c].transform('median'))\n"
        "    bersih[c] = bersih[c].fillna(bersih[c].median())"
    ),
    nbf.v4.new_markdown_cell(
        "Kolom `Klasifikasi Kemiskinan` lama tidak digunakan karena "
        "tidak konsisten dengan nilai kemiskinan."
    ),
    nbf.v4.new_code_cell(
        "fitur = list(batas)\n"
        "print(bersih[fitur].describe().round(2))\n"
        "bersih.drop(columns=['Klasifikasi Kemiskinan']).to_csv('../data/clean/data_bersih.csv', index=False)"
    ),
    nbf.v4.new_markdown_cell("## 3. Analisis eksplorasi"),
    nbf.v4.new_code_cell(
        "prov = (bersih.groupby('Provinsi')[['Kemiskinan','IPM','Pengeluaran','TPT']]\n"
        "         .mean().round(2).sort_values('Kemiskinan', ascending=False))\n"
        "display(prov.head(10))\n"
        "px.histogram(bersih, x='Kemiskinan', nbins=30,\n"
        "             title='Sebaran persentase kemiskinan').show()\n"
        "px.box(bersih, x='Provinsi', y='Kemiskinan',\n"
        "       title='Kemiskinan per provinsi').update_xaxes(tickangle=60).show()"
    ),
    nbf.v4.new_markdown_cell(
        "## 4. Korelasi Spearman dan heatmap\n"
        "Heatmap berikut merangkum arah dan kekuatan hubungan monoton antarvariabel. "
        "Nilai mendekati -1 menunjukkan hubungan berlawanan, nilai mendekati +1 "
        "menunjukkan hubungan searah, dan nilai dekat 0 menunjukkan hubungan monoton yang lemah."
    ),
    nbf.v4.new_code_cell(
        "corr = bersih[fitur].corr(method='spearman').round(2)\n"
        "heatmap = px.imshow(\n"
        "    corr,\n"
        "    text_auto='.2f',\n"
        "    aspect='auto',\n"
        "    color_continuous_scale='RdBu_r',\n"
        "    zmin=-1,\n"
        "    zmax=1,\n"
        "    color_continuous_midpoint=0,\n"
        "    title='Heatmap Korelasi Spearman antarindikator',\n"
        "    labels={'x': 'Indikator', 'y': 'Indikator', 'color': 'Spearman ρ'},\n"
        ")\n"
        "heatmap.update_layout(height=720, width=900, xaxis_tickangle=-35)\n"
        "heatmap.show()\n\n"
        "print('Korelasi terhadap kemiskinan:')\n"
        "display(corr['Kemiskinan'].drop('Kemiskinan').sort_values().to_frame('Spearman ρ'))\n"
        "px.scatter(bersih, x='Pengeluaran', y='Kemiskinan', color='Provinsi',\n"
        "           hover_name='KabKota', trendline='ols',\n"
        "           title='Pengeluaran dan kemiskinan').show()"
    ),
    nbf.v4.new_markdown_cell(
        "## 5. Model klasifikasi eksploratif\n"
        "Kuartil tertinggi kemiskinan dijadikan kelas tinggi. "
        "Kolom `Kemiskinan` dikeluarkan dari fitur agar tidak terjadi kebocoran label."
    ),
    nbf.v4.new_code_cell(
        "bersih['Tinggi'] = (bersih['Kemiskinan'] >= bersih['Kemiskinan'].quantile(.75)).astype(int)\n"
        "X = bersih[[c for c in fitur if c != 'Kemiskinan']]\n"
        "y = bersih['Tinggi']\n"
        "Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)\n"
        "model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)\n"
        "model.fit(Xtr, ytr)\n"
        "print(classification_report(yte, model.predict(Xte), digits=3))\n"
        "penting = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)\n"
        "px.bar(penting, title='Variabel paling berpengaruh pada model').show()"
    ),
    nbf.v4.new_markdown_cell(
        "## 6. Batasan\n"
        "Hasil ini bersifat deskriptif dan model hanya untuk eksplorasi; "
        "korelasi bukan sebab-akibat dan imputasi tidak menggantikan verifikasi sumber."
    ),
]
nbf.write(nb, ROOT / "notebook/analisis_kemiskinan.ipynb")

anomaly_lines = "\n".join(
    f"| {col} | {int(count)} |"
    for col, count in anomali_counts.items()
    if count > 0
)
top_table = "\n".join(
    f"| {row.Provinsi} | {row.KabKota} | {row.Kemiskinan:.2f}% |"
    for row in top_area.itertuples()
)
feature_lines = "\n".join(
    f"| {feature} | {importance:.3f} |"
    for feature, importance in penting.head(5).items()
)

readme = f"""# Analisis Kemiskinan Kabupaten/Kota

Analisis ini menjawab: faktor apa yang paling berkaitan dengan kemiskinan, wilayah mana yang memiliki tingkat kemiskinan tertinggi, dan apakah pola tersebut dapat dimodelkan secara eksploratif.

## Hasil Utama

- Dataset mentah berukuran **{raw_shape[0]} baris × {raw_shape[1]} kolom** dan memiliki **{exact_duplicates} baris duplikat persis**.
- Setelah penghapusan duplikat, **{rows_with_anomaly} baris** mengandung setidaknya satu nilai yang melanggar batas yang ditentukan; ini sekitar **{rows_with_anomaly / len(df) * 100:.1f}%** dari data duplikat-telah-dihapus.
- Korelasi Spearman Pengeluaran–Kemiskinan adalah **{corr.loc['Pengeluaran','Kemiskinan']:.2f}** dan IPM–Kemiskinan adalah **{corr.loc['IPM','Kemiskinan']:.2f}**.
- Model Random Forest eksploratif memperoleh akurasi **{accuracy:.3f}** dan F1 kelas kemiskinan tinggi **{f1_tinggi:.3f}**.

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
{anomaly_lines}

## Temuan Wilayah

| Provinsi | Kabupaten/Kota | Kemiskinan |
|---|---|---:|
{top_table}

## Korelasi dan Interpretasi

Spearman dipilih karena distribusi kemiskinan tidak normal dan beberapa variabel masih mengandung pencilan. Heatmap korelasi tersedia di notebook serta dashboard. Nilai mendekati -1 menunjukkan hubungan monoton berlawanan, nilai dekat +1 menunjukkan hubungan searah, dan nilai dekat 0 menunjukkan hubungan monoton yang lemah. Korelasi menunjukkan hubungan, bukan sebab-akibat.

## Model Eksploratif

Label dibuat dari kuartil tertinggi kemiskinan. Kolom kemiskinan tidak dimasukkan sebagai fitur.

| Variabel | Tingkat pentingan |
|---|---:|
{feature_lines}

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
"""
(ROOT / "README.md").write_text(readme, encoding="utf-8")

print({
    "raw_shape": raw_shape,
    "exact_duplicates": exact_duplicates,
    "rows_with_anomaly": rows_with_anomaly,
    "accuracy": accuracy,
    "f1_tinggi": f1_tinggi,
})
