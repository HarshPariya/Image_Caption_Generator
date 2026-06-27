"""
app.py  –  VisionCaption AI
============================
Self-contained Streamlit app frontend. 
Backend logic is imported from the `utils` module.

Run:
    streamlit run app.py
"""

import os
import random
import streamlit as st

# ── Suppress TensorFlow noise ─────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="VisionCaption AI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Import backend modules ───────────────────────────────────────────────────
from utils.config import MODELS_READY, DEMO_CAPTIONS, IMAGE_SIZE
from utils.preprocessing import validate_and_load_image, prepare_image_for_model
from utils.feature_extractor import load_inception_v3, extract_features
from utils.caption_generator import load_caption_model, load_tokenizer, generate_caption

# ─────────────────────────────────────────────────────────────────────────────
# @st.cache_resource — loaded ONCE, reused across all reruns / sessions
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading InceptionV3 feature extractor…")
def _get_extractor():
    return load_inception_v3()

@st.cache_resource(show_spinner="⏳ Loading caption model & tokenizer…")
def _get_caption_assets():
    return load_caption_model(), load_tokenizer()

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, .stApp {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background: #eef2f7 !important;
}
.block-container {
    max-width: 760px !important;
    padding: 0 1.25rem 4rem !important;
    margin: 0 auto !important;
}
#MainMenu, footer, header { visibility: hidden !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #93c5fd !important;
    border-radius: 14px !important;
    background: #f0f6ff !important;
    padding: 0.5rem 1rem !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #64748b !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 100%) !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 4px 18px rgba(29,78,216,0.30) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af 0%, #0284c7 100%) !important;
    box-shadow: 0 8px 28px rgba(29,78,216,0.40) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Spinner */
.stSpinner > div > div { border-top-color: #1d4ed8 !important; }

/* Image */
[data-testid="stImage"] img {
    border-radius: 14px !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.09) !important;
    max-height: 460px !important;
    object-fit: contain !important;
    width: 100% !important;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    background: #fff !important;
    margin-top: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.85rem !important;
    color: #475569 !important;
}

/* white card helper class used inline */
.vc-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.4rem 1.5rem 1.25rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [("caption", None), ("cap_mode", None), ("last_file", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1e3a8a 0%,#1d4ed8 55%,#0ea5e9 100%);
    border-radius:22px;padding:2.5rem 2rem 2.25rem;margin:1.25rem 0 1.25rem;
    text-align:center;box-shadow:0 10px 40px rgba(29,78,216,0.22);
    position:relative;overflow:hidden;">
  <div style="position:absolute;top:-55px;right:-55px;width:170px;height:170px;
    border-radius:50%;background:rgba(255,255,255,0.05);pointer-events:none;"></div>
  <div style="position:absolute;bottom:-35px;left:-35px;width:120px;height:120px;
    border-radius:50%;background:rgba(255,255,255,0.05);pointer-events:none;"></div>

  <div style="display:inline-flex;align-items:center;justify-content:center;
    width:68px;height:68px;border-radius:20px;margin-bottom:1rem;
    background:rgba(255,255,255,0.14);border:1px solid rgba(255,255,255,0.28);
    font-size:2.2rem;">🤖</div>

  <h1 style="color:#fff;font-size:2.5rem;font-weight:800;letter-spacing:-0.03em;
    margin:0 0 0.6rem;line-height:1.1;">
    VisionCaption <span style="color:#7dd3fc;">AI</span>
  </h1>
  <p style="color:rgba(255,255,255,0.80);font-size:0.97rem;line-height:1.65;
    max-width:460px;margin:0 auto 1.3rem;">
    Upload any photo and our deep-learning model instantly describes what it sees —
    powered by <strong style="color:#fff;">InceptionV3 + LSTM</strong>.
  </p>
  <div style="display:flex;justify-content:center;gap:0.55rem;flex-wrap:wrap;">
    <span style="background:rgba(255,255,255,0.14);color:#e0f2fe;
      border:1px solid rgba(255,255,255,0.22);padding:4px 14px;
      border-radius:999px;font-size:0.75rem;font-weight:600;">⚡ InceptionV3 Features</span>
    <span style="background:rgba(255,255,255,0.14);color:#e0f2fe;
      border:1px solid rgba(255,255,255,0.22);padding:4px 14px;
      border-radius:999px;font-size:0.75rem;font-weight:600;">🧠 LSTM Decoder</span>
    <span style="background:rgba(255,255,255,0.14);color:#e0f2fe;
      border:1px solid rgba(255,255,255,0.22);padding:4px 14px;
      border-radius:999px;font-size:0.75rem;font-weight:600;">📊 ~8K Vocabulary</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STATUS BANNER
# ─────────────────────────────────────────────────────────────────────────────
if MODELS_READY:
    st.markdown("""
<div style="background:#f0fdf4;border:1px solid #86efac;border-left:4px solid #16a34a;
    border-radius:12px;padding:0.75rem 1.2rem;margin-bottom:1rem;
    display:flex;gap:10px;align-items:center;">
  <span style="font-size:1.1rem;">✅</span>
  <p style="font-weight:600;color:#15803d;margin:0;font-size:0.86rem;">
    Model ready — real AI captioning is active.
  </p>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<div style="background:#fffbeb;border:1px solid #fcd34d;border-left:4px solid #f59e0b;
    border-radius:12px;padding:0.75rem 1.2rem;margin-bottom:1rem;
    display:flex;gap:10px;align-items:flex-start;">
  <span style="font-size:1.1rem;flex-shrink:0;margin-top:1px;">⚠️</span>
  <div>
    <p style="font-weight:700;color:#92400e;margin:0 0 2px;font-size:0.86rem;">Demo Mode</p>
    <p style="color:#b45309;margin:0;font-size:0.78rem;">
      Place <code style="background:#fef3c7;border-radius:4px;padding:1px 5px;
      font-size:0.76rem;color:#92400e;">image_caption_model.keras</code> and 
      <code style="background:#fef3c7;border-radius:4px;padding:1px 5px;
      font-size:0.76rem;color:#92400e;">tokenizer.pkl</code> in the 
      <code style="background:#fef3c7;border-radius:4px;padding:1px 5px;
      font-size:0.76rem;color:#92400e;">models/</code> folder for real AI captions.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────────────────────
s1, s2, s3 = st.columns(3)
for col, val, lbl in [
    (s1, "InceptionV3", "Architecture"),
    (s2, "2048", "Feature Dim"),
    (s3, "~8.8K", "Vocab Size"),
]:
    with col:
        st.markdown(f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;
    padding:1rem 0.75rem;text-align:center;
    box-shadow:0 1px 6px rgba(0,0,0,0.045);">
  <p style="font-size:1.1rem;font-weight:800;color:#1d4ed8;margin:0 0 3px;">{val}</p>
  <p style="font-size:0.68rem;color:#94a3b8;font-weight:600;margin:0;
    text-transform:uppercase;letter-spacing:0.07em;">{lbl}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:0.85rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LIMITATIONS (Explanation for users)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("ℹ️  Why might the AI generate a 'wrong' or weird caption?"):
    st.markdown("""
    <div style="font-size:0.9rem; color:#334155; line-height:1.6;">
    <strong>This is not a bug! It's a limitation of the training data.</strong><br><br>
    This AI was trained from scratch on the <strong>Flickr8k dataset</strong>, which contains 8,000 pictures mostly consisting of people, children, dogs, and grassy fields. <br><br>
    <ul>
        <li><strong>Limited World Knowledge:</strong> The dataset contains <em>zero</em> pictures of the moon, outer space, or complex cityscapes. </li>
        <li><strong>Forced Guessing:</strong> When the model sees an image it doesn't recognize (like the moon against a dark sky), it searches its memory for the closest shape it knows. A bright shape in the sky might look like a frisbee or a person jumping, leading to forced guesses like <em>"The girl is flying through the air."</em></li>
    </ul>
    <strong>How to test the model properly:</strong><br>
    Upload images similar to its training data:
    <ul>
        <li>🐕 A dog running on grass</li>
        <li>🚲 A person riding a bike</li>
        <li>🏃‍♂️ A child playing outside</li>
    </ul>
    To recognize everything in the world, the model would need to be trained on a massive dataset like MS COCO (300,000+ images).
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:0.85rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD CARD HEADER  (self-contained)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;
    padding:1.25rem 1.5rem 0.85rem;
    box-shadow:0 2px 10px rgba(0,0,0,0.04);margin-bottom:4px;">
  <div style="display:flex;align-items:center;gap:11px;">
    <div style="width:38px;height:38px;border-radius:11px;flex-shrink:0;
      background:linear-gradient(135deg,#dbeafe,#bfdbfe);
      display:flex;align-items:center;justify-content:center;font-size:1.1rem;">📂</div>
    <div>
      <p style="font-weight:700;color:#1a202c;font-size:0.95rem;margin:0 0 1px;">
        Upload Your Image</p>
      <p style="color:#94a3b8;font-size:0.77rem;margin:0;">
        Supports JPG · JPEG · PNG — max 10 MB</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOADER (native widget)
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    label="Upload your image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    key="uploader",
)
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN FLOW
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:

    # Reset when a new file is chosen
    if st.session_state.last_file != uploaded_file.name:
        st.session_state.caption  = None
        st.session_state.cap_mode = None
        st.session_state.last_file = uploaded_file.name

    try:
        # Preprocessing: validate and load
        image = validate_and_load_image(uploaded_file)
        w, h    = image.size
        size_kb = round(uploaded_file.size / 1024, 1)
        ext     = uploaded_file.name.rsplit(".", 1)[-1].upper()

        # ── IMAGE PREVIEW CARD ────────────────────────────────────────────────
        st.markdown(f"""
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;
        padding:1.4rem 1.5rem 0.75rem;margin-bottom:0px;
        box-shadow:0 2px 10px rgba(0,0,0,0.05);">
      <div style="display:flex;align-items:center;gap:11px;margin-bottom:0.85rem;">
        <div style="width:38px;height:38px;border-radius:11px;flex-shrink:0;
          background:linear-gradient(135deg,#dbeafe,#bfdbfe);
          display:flex;align-items:center;justify-content:center;font-size:1.1rem;">🖼️</div>
        <div>
          <p style="font-weight:700;color:#1a202c;font-size:0.95rem;margin:0 0 1px;">
            Image Preview</p>
          <p style="color:#94a3b8;font-size:0.77rem;margin:0;">
            {uploaded_file.name}&nbsp;·&nbsp;{w}&nbsp;×&nbsp;{h}&nbsp;px</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

        # Native st.image
        st.image(image, width="stretch")

        # Metadata badges
        st.markdown(f"""
    <div style="background:#fff;border:1px solid #e2e8f0;border-top:none;
        border-radius:0 0 18px 18px;padding:0.75rem 1.5rem 1.1rem;
        margin-bottom:0.75rem;box-shadow:0 2px 10px rgba(0,0,0,0.05);">
      <div style="display:flex;flex-wrap:wrap;gap:0.45rem;">
        <span style="background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;
          border-radius:999px;padding:3px 12px;font-size:0.72rem;font-weight:600;">
          {ext} file</span>
        <span style="background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;
          border-radius:999px;padding:3px 12px;font-size:0.72rem;font-weight:600;">
          {w}&nbsp;×&nbsp;{h}&nbsp;px</span>
        <span style="background:#fdf4ff;color:#7c3aed;border:1px solid #e9d5ff;
          border-radius:999px;padding:3px 12px;font-size:0.72rem;font-weight:600;">
          {size_kb}&nbsp;KB</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

        # ── GENERATE CAPTION CARD ─────────────────────────────────────────────
        st.markdown("""
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;
        padding:1.4rem 1.5rem 1rem;margin-bottom:0.5rem;
        box-shadow:0 2px 10px rgba(0,0,0,0.05);">
      <div style="display:flex;align-items:center;gap:11px;margin-bottom:0.85rem;">
        <div style="width:38px;height:38px;border-radius:11px;flex-shrink:0;
          background:linear-gradient(135deg,#dbeafe,#bfdbfe);
          display:flex;align-items:center;justify-content:center;font-size:1.1rem;">⚡</div>
        <div>
          <p style="font-weight:700;color:#1a202c;font-size:0.95rem;margin:0 0 1px;">
            Generate Caption</p>
          <p style="color:#94a3b8;font-size:0.77rem;margin:0;">
            Click the button below to run AI inference on your image</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

        # ── BUTTON (always active) ─────────────────────────────────────────────
        if st.button("✨  Generate Caption", key="gen_btn"):
            st.session_state.caption  = None
            st.session_state.cap_mode = None

            if MODELS_READY:
                with st.spinner("🔍  Extracting features & generating caption…"):
                    try:
                        # 1. Load models (cached)
                        extractor = _get_extractor()
                        lstm, tok = _get_caption_assets()
                        
                        # 2. Preprocess & extract features
                        arr = prepare_image_for_model(image, IMAGE_SIZE)
                        feats = extract_features(extractor, arr)
                        
                        # 3. Generate caption
                        caption = generate_caption(lstm, tok, feats)
                        
                        st.session_state.caption  = caption
                        st.session_state.cap_mode = "real"
                    except Exception as exc:
                        st.session_state.caption  = str(exc)
                        st.session_state.cap_mode = "error"
            else:
                with st.spinner("✨  Generating demo caption…"):
                    import time; time.sleep(0.8)
                    st.session_state.caption  = random.choice(DEMO_CAPTIONS)
                    st.session_state.cap_mode = "demo"

        # ── RESULT ─────────────────────────────────────────────────────────────
        if st.session_state.caption and st.session_state.cap_mode:
            mode = st.session_state.cap_mode
            cap  = st.session_state.caption

            if mode == "error":
                # Flat HTML — no nested divs
                st.markdown(f"""
    <div style="background:#fef2f2;border:1px solid #fecaca;border-left:4px solid #ef4444;
        border-radius:14px;padding:1.1rem 1.25rem;margin-top:0.8rem;">
      <p style="font-weight:700;color:#b91c1c;margin:0 0 6px;font-size:0.9rem;">❌ Inference Error</p>
      <p style="color:#dc2626;margin:0;font-size:0.83rem;word-break:break-word;">{cap}</p>
    </div>
    """, unsafe_allow_html=True)

            else:
                # ── Result card: flat HTML only, no nested divs ────────────────
                demo_label = ""
                if mode == "demo":
                    demo_label = (
                        '<span style="background:#fef3c7;color:#92400e;'
                        'border:1px solid #fcd34d;border-radius:6px;padding:2px 9px;'
                        'font-size:0.67rem;font-weight:700;text-transform:uppercase;'
                        'letter-spacing:0.06em;margin-right:8px;">Demo Mode</span>'
                    )

                st.markdown(f"""
    <div style="background:linear-gradient(135deg,#eff6ff 0%,#dbeafe 100%);
        border:1px solid #bfdbfe;border-left:4px solid #1d4ed8;
        border-radius:16px;padding:1.4rem 1.6rem;margin-top:0.85rem;">
      <p style="margin:0 0 6px;">{demo_label}
        <span style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
          text-transform:uppercase;color:#1d4ed8;">✨ Generated Caption</span>
      </p>
      <p style="font-size:1.15rem;font-weight:600;color:#1e3a8a;
        line-height:1.75;margin:0;font-style:italic;">
        &#8220;{cap}&#8221;
      </p>
    </div>
    """, unsafe_allow_html=True)

                with st.expander("📋  Copy caption text"):
                    st.code(cap, language=None)

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    except ValueError as e:
        st.markdown(f"""
        <div style="background:#fef2f2;border:1px solid #fecaca;border-left:4px solid #ef4444;
            border-radius:14px;padding:1.1rem 1.25rem;margin-top:0.8rem;">
          <p style="font-weight:700;color:#b91c1c;margin:0 0 6px;font-size:0.9rem;">❌ Validation Error</p>
          <p style="color:#dc2626;margin:0;font-size:0.83rem;word-break:break-word;">{str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
        
# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;
    padding:3rem 1.5rem;text-align:center;
    box-shadow:0 2px 10px rgba(0,0,0,0.04);">
  <div style="display:inline-flex;align-items:center;justify-content:center;
    width:80px;height:80px;border-radius:22px;
    background:linear-gradient(135deg,#dbeafe,#eff6ff);
    border:2.5px dashed #93c5fd;font-size:2.2rem;margin-bottom:1.1rem;">🖼️</div>
  <p style="font-weight:700;color:#334155;font-size:1.05rem;margin:0 0 6px;">
    No image uploaded yet</p>
  <p style="color:#94a3b8;font-size:0.85rem;margin:0 auto;
    max-width:280px;line-height:1.6;">
    Drag and drop or click above to select a JPG / PNG photo.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:2rem;padding-top:1.25rem;
    border-top:1px solid #e2e8f0;">
  <p style="color:#94a3b8;font-size:0.77rem;margin:0;">
    Built with <span style="color:#ef4444;">♥</span> using
    <strong style="color:#64748b;">Streamlit</strong> ·
    <strong style="color:#64748b;">TensorFlow / Keras</strong> ·
    <strong style="color:#64748b;">InceptionV3 + LSTM</strong>
  </p>
</div>
""", unsafe_allow_html=True)
