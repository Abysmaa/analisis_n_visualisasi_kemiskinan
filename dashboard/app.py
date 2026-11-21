from pathlib import Path

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
