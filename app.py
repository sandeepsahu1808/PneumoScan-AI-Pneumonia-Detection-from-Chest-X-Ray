import os
import sys
import json
import gdown
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Ensure src modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.gradcam import get_gradcam_heatmap, overlay_heatmap

# Prevent Mac GPU (MPS) from freezing during Streamlit inference
tf.config.set_visible_devices([], 'GPU')

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* ── Background ── */
.stApp, .main, .block-container {
    background: #060B18 !important;
    color: #E2E8F0 !important;
}
.block-container {
    padding: 2rem 4rem !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Typography ── */
h1, h2, h3, h4, p, span, label, div { color: #E2E8F0 !important; }

/* ── HERO HEADER ── */
.hero-wrapper {
    background: linear-gradient(135deg, #0F172A 0%, #0D1B3E 50%, #0F172A 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 20px;
    padding: 48px 56px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(37,99,235,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-wrapper::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 400px; height: 200px;
    background: radial-gradient(ellipse, rgba(99,179,237,0.08) 0%, transparent 70%);
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(37,99,235,0.15);
    border: 1px solid rgba(37,99,235,0.4);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #60A5FA !important;
    margin-bottom: 20px;
}
.hero-title {
    font-size: 3.6rem !important;
    font-weight: 900 !important;
    line-height: 1.1 !important;
    background: linear-gradient(135deg, #FFFFFF 0%, #93C5FD 60%, #60A5FA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 16px;
}
.hero-sub {
    color: #94A3B8 !important;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 32px;
    max-width: 520px;
}
.pill-row { display: flex; gap: 12px; flex-wrap: wrap; }
.stat-pill {
    display: flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 10px 18px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #E2E8F0 !important;
}
.stat-pill .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #22C55E;
    box-shadow: 0 0 8px #22C55E;
    display: inline-block;
}
.stat-pill .val { color: #60A5FA !important; font-weight: 700; }

/* ── UPLOAD CARD ── */
.upload-card {
    background: #0D1B3E;
    border: 2px dashed rgba(99,179,237,0.3);
    border-radius: 16px;
    padding: 48px 32px;
    text-align: center;
    margin-bottom: 32px;
    transition: border-color 0.3s;
}
.upload-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #E2E8F0 !important;
    margin-bottom: 6px;
}
.upload-sub {
    color: #64748B !important;
    font-size: 0.85rem;
}

/* ── RESULT PANEL ── */
.result-panel {
    background: #0D1B3E;
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 24px;
}
.panel-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748B !important;
    margin-bottom: 14px;
}

/* ── DIAGNOSIS BADGE ── */
.badge-pneumonia {
    background: linear-gradient(135deg, #7F1D1D, #991B1B);
    border: 1px solid #EF4444;
    border-radius: 16px;
    padding: 20px 32px;
    text-align: center;
    box-shadow: 0 0 40px rgba(239,68,68,0.25);
}
.badge-normal {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22C55E;
    border-radius: 16px;
    padding: 20px 32px;
    text-align: center;
    box-shadow: 0 0 40px rgba(34,197,94,0.2);
}
.badge-title {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 600;
    opacity: 0.7;
    margin-bottom: 6px;
}
.badge-label {
    font-size: 2rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.5px;
}
.badge-pneumonia .badge-label { color: #FCA5A5 !important; }
.badge-normal .badge-label    { color: #86EFAC !important; }
.badge-confidence {
    font-size: 0.85rem;
    opacity: 0.7;
    margin-top: 4px;
}

/* ── CONFIDENCE BAR ── */
.conf-wrap { margin-top: 20px; }
.conf-label {
    display: flex; justify-content: space-between;
    font-size: 0.8rem; font-weight: 600;
    color: #94A3B8 !important;
    margin-bottom: 8px;
}
.conf-bar-bg {
    height: 10px;
    background: rgba(255,255,255,0.08);
    border-radius: 100px;
    overflow: hidden;
}
.conf-bar-fill-p { background: linear-gradient(90deg, #EF4444, #F87171); border-radius: 100px; height: 100%; }
.conf-bar-fill-n { background: linear-gradient(90deg, #16A34A, #4ADE80); border-radius: 100px; height: 100%; }
.threshold-tag {
    margin-top: 10px;
    font-size: 0.73rem;
    color: #475569 !important;
    text-align: right;
}

/* ── METRICS TABLE ── */
.metrics-header {
    font-size: 1.15rem;
    font-weight: 800;
    color: #E2E8F0 !important;
    margin-bottom: 4px;
}
.metrics-sub {
    font-size: 0.82rem;
    color: #64748B !important;
    margin-bottom: 24px;
}
.metric-card {
    background: #0F172A;
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 14px;
    padding: 24px 28px;
}
.metric-card-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #60A5FA !important;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.metric-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.metric-row:last-child { border-bottom: none; }
.metric-name { font-size: 0.85rem; color: #94A3B8 !important; }
.metric-value { font-size: 0.95rem; font-weight: 700; color: #E2E8F0 !important; }
.metric-value.green { color: #4ADE80 !important; }
.metric-value.blue  { color: #60A5FA !important; }

/* ── FOOTER ── */
.footer {
    text-align: center;
    padding: 36px 0 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 40px;
}
.footer-text { font-size: 0.8rem; color: #334155 !important; }
.footer-link { color: #2563EB !important; text-decoration: none; }

/* Override Streamlit file uploader — hide label completely */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] > div > label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploaderDropzone"] + div {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(99,179,237,0.25) !important;
    border-radius: 14px !important;
    padding: 24px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── MODEL WEIGHTS (Google Drive) ────────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.h5')
_GDRIVE_FILE_ID = os.getenv('GDRIVE_MODEL_FILE_ID', 'YOUR_FILE_ID')

if not os.path.exists(_MODEL_PATH) and _GDRIVE_FILE_ID != 'YOUR_FILE_ID':
    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    gdown.download(
        f'https://drive.google.com/uc?id={_GDRIVE_FILE_ID}',
        _MODEL_PATH,
        quiet=False,
    )

# ─── MODEL LOADING ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_pneumonia_model():
    model_path = _MODEL_PATH
    if os.path.exists(model_path):
        return load_model(model_path, compile=False)
    return None

model = load_pneumonia_model()
OPTIMAL_THRESHOLD = 0.4665

# ─── HERO HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-tag">🫁 &nbsp; AI-Powered Radiology Assistant</div>
    <div class="hero-title">PneumoScan AI</div>
    <div class="hero-sub">Upload a chest X-ray and get an instant AI diagnosis with visual Grad-CAM explanations — built on ResNet50 and trained on 10,000+ clinical images.</div>
    <div class="pill-row">
        <div class="stat-pill"><span class="dot"></span> <span class="val">93.0%</span> Accuracy</div>
        <div class="stat-pill"><span class="dot"></span> <span class="val">0.98</span> AUC-ROC</div>
        <div class="stat-pill"><span class="dot"></span> <span class="val">93.0%</span> Recall (Pneumonia)</div>
        <div class="stat-pill"><span class="dot"></span> <span class="val">10k+</span> Training Images</div>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️  Model not found at `models/best_model.h5`. Please complete training first.")
    st.stop()

# ─── UPLOAD SECTION ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["png", "jpg", "jpeg"],
    help="Supports JPG, JPEG, PNG",
    label_visibility="collapsed"
)
st.markdown('<div class="upload-sub" style="text-align:center; margin-top: -8px; margin-bottom: 32px;">Drag &amp; drop or click to upload a chest X-ray · PNG, JPG, JPEG</div>', unsafe_allow_html=True)

# ─── RESULTS ──────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = img_to_array(img_resized) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    # Predict
    pred_prob = float(model.predict(img_array_exp, verbose=0)[0][0])
    is_pneumonia = pred_prob >= OPTIMAL_THRESHOLD
    confidence = pred_prob if is_pneumonia else (1 - pred_prob)

    # Grad-CAM
    heatmap_img = None
    try:
        heatmap = get_gradcam_heatmap(model, img_array_exp, last_conv_layer_name='conv5_block3_out')
        heatmap_img = overlay_heatmap(heatmap, np.array(image.resize((224, 224))) / 255.0)
    except Exception:
        pass

    # ── Image columns ──
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="panel-label">Original X-Ray</div>', unsafe_allow_html=True)
        st.image(image, width='stretch')
    with col2:
        st.markdown('<div class="panel-label">Grad-CAM — Model Focus Area</div>', unsafe_allow_html=True)
        if heatmap_img is not None:
            st.image(heatmap_img, width='stretch')
        else:
            st.warning("Grad-CAM unavailable — showing original image")
            st.image(image, width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Diagnosis badge ──
    badge_class = "badge-pneumonia" if is_pneumonia else "badge-normal"
    label_text  = "PNEUMONIA DETECTED" if is_pneumonia else "NORMAL — NO PNEUMONIA"
    bar_class   = "conf-bar-fill-p" if is_pneumonia else "conf-bar-fill-n"
    conf_pct    = confidence * 100
    bar_width   = f"{conf_pct:.1f}%"

    st.markdown(f"""
    <div class="{badge_class}">
        <div class="badge-title">Diagnosis Result</div>
        <div class="badge-label">{label_text}</div>
        <div class="badge-confidence">Confidence: {conf_pct:.2f}%</div>
    </div>
    <div class="conf-wrap">
        <div class="conf-label"><span>Confidence Score</span><span>{conf_pct:.1f}%</span></div>
        <div class="conf-bar-bg"><div class="{bar_class}" style="width:{bar_width}"></div></div>
        <div class="threshold-tag">Optimal threshold: {OPTIMAL_THRESHOLD} · Raw probability: {pred_prob:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color:rgba(255,255,255,0.06)'><br>", unsafe_allow_html=True)

# ─── METRICS SECTION ──────────────────────────────────────────────────────────────
metrics_path = os.path.join(os.path.dirname(__file__), 'results', 'metrics.json')
if os.path.exists(metrics_path):
    try:
        with open(metrics_path) as f:
            mx = json.load(f)
        d = mx.get('threshold_default', {})
        o = mx.get('threshold_optimal', {})
        ds = mx.get('dataset', {})

        st.markdown('<div class="metrics-header">Model Performance on Unseen Test Set</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metrics-sub">Evaluated on {ds.get("test_samples", "—")} images · {ds.get("normal_samples","—")} Normal &nbsp;|&nbsp; {ds.get("pneumonia_samples","—")} Pneumonia</div>', unsafe_allow_html=True)

        def metric_row(name, val, cls="blue"):
            return f'<div class="metric-row"><span class="metric-name">{name}</span><span class="metric-value {cls}">{val}</span></div>'

        cm1, cm2 = st.columns(2, gap="large")
        with cm1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Default Threshold — 0.5000</div>
                {metric_row("Accuracy",  f"{d.get('accuracy',0)*100:.1f}%")}
                {metric_row("Precision", f"{d.get('precision',0)*100:.1f}%")}
                {metric_row("Recall",    f"{d.get('recall',0)*100:.1f}%", "green")}
                {metric_row("F1-Score",  f"{d.get('f1',0):.4f}")}
                {metric_row("AUC-ROC",   f"{d.get('auc_roc',0):.4f}", "green")}
            </div>
            """, unsafe_allow_html=True)
        with cm2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Optimal Threshold — {o.get('threshold', OPTIMAL_THRESHOLD):.4f}</div>
                {metric_row("Accuracy",  f"{o.get('accuracy',0)*100:.1f}%")}
                {metric_row("Precision", f"{o.get('precision',0)*100:.1f}%")}
                {metric_row("Recall",    f"{o.get('recall',0)*100:.1f}%", "green")}
                {metric_row("F1-Score",  f"{o.get('f1',0):.4f}")}
                {metric_row("AUC-ROC",   f"{o.get('auc_roc',0):.4f}", "green")}
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        pass

# ─── FOOTER ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-text">
        Built with TensorFlow &nbsp;·&nbsp; ResNet50 &nbsp;·&nbsp; Grad-CAM &nbsp;·&nbsp; Streamlit<br><br>
        <a class="footer-link" href="https://github.com/sandeepsahu1808" target="_blank">GitHub</a>
    </div>
</div>
""", unsafe_allow_html=True)
