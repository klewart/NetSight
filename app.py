"""
app.py  -  NetSight Road Network Resilience Dashboard
=====================================================
Streamlit presentation layer for the full NetSight pipeline.

    Upload satellite tile  ->  ViT-UNet inference (CPU)
    ->  skeleton + graph extraction  ->  cascade failure simulation
    ->  resilience index dashboard

Run:  streamlit run app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import streamlit as st
import torch
from PIL import Image

# -- Path injection so we can import project modules --
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "my_agents"))

from src.model import ViTUNet
from src.mask_to_graph import mask_to_graph

# -- Constants ----------------------------------------------------------------
_INPUT_SIZE = 512           # Must be divisible by 16 (4x MaxPool2d stride-2)
_DEFAULT_THRESHOLD = 0.50
_CASCADE_FRACTION = 0.05    # Destroy top 5 % of edges by betweenness
_BOTTLENECK_FRACTION = 0.10 # Highlight top 10 % post-failure bottlenecks

# -- Page configuration -------------------------------------------------------
st.set_page_config(
    page_title="NetSight | Road Network Resilience",
    page_icon="\U0001F6F0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
/* ── Import a clean geometric font ──────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero header gradient ───────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00d4ff, #a855f7, #ec4899, #00d4ff);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer { to { background-position: -200% 0; } }
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
}

/* ── Glass-morphism metric card ─────────────────────────────────────── */
.glass-card {
    background: rgba(17, 24, 39, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.3s;
}
.glass-card:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 212, 255, 0.3);
}
.glass-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.glass-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e2e8f0;
}
.glass-value.cyan   { color: #00d4ff; }
.glass-value.green  { color: #22c55e; }
.glass-value.amber  { color: #f59e0b; }
.glass-value.rose   { color: #f43f5e; }
.glass-value.purple { color: #a855f7; }
.glass-delta {
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.3rem;
}

/* ── Resilience gauge bar ───────────────────────────────────────────── */
.gauge-outer {
    width: 100%;
    height: 18px;
    background: #1e293b;
    border-radius: 9px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
}
.gauge-fill {
    height: 100%;
    border-radius: 9px;
    transition: width 0.8s ease;
}

/* ── Cascade button pulse ───────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    animation: pulse-border 2s ease-in-out infinite;
}
@keyframes pulse-border {
    0%, 100% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.4); }
    50%      { box-shadow: 0 0 0 8px rgba(244, 63, 94, 0); }
}

/* ── Section divider ────────────────────────────────────────────────── */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL LOADING  (cached once across all Streamlit reruns)
# =============================================================================
@st.cache_resource(show_spinner="Loading ViT-UNet weights ...")
def _load_model() -> ViTUNet:
    model = ViTUNet(base_ch=64, vit_depth=4, vit_heads=8)
    w_path = ROOT / "my_agents" / "weights" / "model_weights.pth"
    if not w_path.exists():
        st.error(f"Model weights not found at `{w_path}`")
        st.stop()
    state = torch.load(str(w_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# =============================================================================
# IMAGE PRE-PROCESSING
# =============================================================================
def _preprocess(image: Image.Image) -> np.ndarray:
    """Resize + pad to (_INPUT_SIZE x _INPUT_SIZE) and return float32 [0,1] RGB array."""
    img = image.convert("RGB")
    w, h = img.size
    scale = _INPUT_SIZE / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    padded = Image.new("RGB", (_INPUT_SIZE, _INPUT_SIZE), (0, 0, 0))
    padded.paste(img, ((_INPUT_SIZE - new_w) // 2, (_INPUT_SIZE - new_h) // 2))
    return np.array(padded, dtype=np.float32) / 255.0


# =============================================================================
# INFERENCE
# =============================================================================
def _run_inference(model: ViTUNet,
                   img_array: np.ndarray,
                   threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (probability_map, binary_mask) from a float32 HWC image."""
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask = (probs > threshold).astype(np.uint8)
    return probs, mask


# =============================================================================
# MATPLOTLIB HELPERS  (dark-theme figures)
# =============================================================================
_FIG_BG = "#0a0e17"
_AX_BG  = "#0a0e17"

# Colour palette
_C_EDGE      = "#00d4ff"    # cyan  - normal edge
_C_JUNCTION  = "#22c55e"    # green - junction node
_C_ENDPOINT  = "#a855f7"    # purple - endpoint node
_C_REMOVED   = "#f43f5e"    # rose  - destroyed edge
_C_BOTTLENCK = "#f59e0b"    # amber - rerouted bottleneck


def _dark_fig(size=(6, 6)):
    fig, ax = plt.subplots(figsize=size, facecolor=_FIG_BG)
    ax.set_facecolor(_AX_BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return fig, ax


def _plot_original(img: np.ndarray):
    fig, ax = _dark_fig()
    ax.imshow(img)
    ax.set_title("Satellite Input", color="white", fontsize=13,
                 fontweight="bold", pad=10)
    plt.tight_layout()
    return fig


def _plot_mask(img: np.ndarray, probs: np.ndarray, mask: np.ndarray):
    fig, ax = _dark_fig()
    ax.imshow(img)
    overlay = np.ma.masked_where(mask == 0, probs)
    ax.imshow(overlay, cmap="inferno", alpha=0.72, vmin=0.01, vmax=1.0)
    ax.set_title("Road Segmentation Mask", color="white", fontsize=13,
                 fontweight="bold", pad=10)
    plt.tight_layout()
    return fig


def _edge_in(u: int, v: int, edge_set: set | None) -> bool:
    """Check membership ignoring edge direction (undirected)."""
    if edge_set is None:
        return False
    return frozenset((u, v)) in edge_set


def _normalise_edge_set(raw: set | None) -> set | None:
    if raw is None:
        return None
    return {frozenset(e) for e in raw}


def _plot_network(G: nx.Graph,
                  shape: tuple[int, int],
                  skeleton: np.ndarray | None = None,
                  removed_edges: set | None = None,
                  bottleneck_edges: set | None = None,
                  title: str = "Road Network  G(V, E)"):
    """Render the road graph.  Optionally mark removed & bottleneck edges."""
    fig, ax = _dark_fig((7, 7))
    H, W = shape
    removed_f    = _normalise_edge_set(removed_edges)
    bottleneck_f = _normalise_edge_set(bottleneck_edges)

    # Faint skeleton backdrop
    if skeleton is not None:
        bg = np.zeros((H, W, 4), dtype=np.float32)
        bg[skeleton] = [0.18, 0.25, 0.35, 0.25]
        ax.imshow(bg, origin="upper")

    if G.number_of_nodes() == 0:
        ax.text(W / 2, H / 2, "No graph extracted", ha="center",
                va="center", color="#64748b", fontsize=15)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        plt.tight_layout()
        return fig

    # -- Draw edges (using the actual pixel paths for curved roads) ----------
    for u, v, data in G.edges(data=True):
        path = data.get("path", [])
        if len(path) < 2:
            p1, p2 = G.nodes[u]["pos"], G.nodes[v]["pos"]
            path = [p1, p2]
        cols = [int(c) for _, c in path]
        rows = [int(r) for r, _ in path]

        is_removed = _edge_in(u, v, removed_f)
        is_bottle  = _edge_in(u, v, bottleneck_f)

        if is_removed:
            ax.plot(cols, rows, color=_C_REMOVED, lw=2.8, alpha=0.85,
                    linestyle=(0, (5, 4)), zorder=3)
        elif is_bottle:
            ax.plot(cols, rows, color=_C_BOTTLENCK, lw=3.2, alpha=0.92, zorder=4)
        else:
            ax.plot(cols, rows, color=_C_EDGE, lw=1.4, alpha=0.55, zorder=2)

    # -- Draw nodes ----------------------------------------------------------
    for nid, data in G.nodes(data=True):
        r, c = int(data["pos"][0]), int(data["pos"][1])
        if data.get("kind") == "junction":
            ax.plot(c, r, "o", color=_C_JUNCTION, ms=9,
                    markeredgecolor="white", markeredgewidth=1.4, zorder=6)
        else:
            ax.plot(c, r, "s", color=_C_ENDPOINT, ms=7,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=6)

    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)

    # -- Legend --------------------------------------------------------------
    handles = [
        Line2D([], [], marker="o", color="w", markerfacecolor=_C_JUNCTION,
               ms=8, ls="None", label="Junction"),
        Line2D([], [], marker="s", color="w", markerfacecolor=_C_ENDPOINT,
               ms=7, ls="None", label="Endpoint"),
        Line2D([], [], color=_C_EDGE, lw=2, label="Road Segment"),
    ]
    if removed_f:
        handles.append(Line2D([], [], color=_C_REMOVED, lw=2,
                              ls="--", label="Destroyed"))
    if bottleneck_f:
        handles.append(Line2D([], [], color=_C_BOTTLENCK, lw=3,
                              label="Bottleneck"))
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              facecolor="#111827", edgecolor="#1e293b", labelcolor="white")
    plt.tight_layout()
    return fig


# =============================================================================
# CASCADE FAILURE SIMULATION
# =============================================================================
def _simulate_cascade(G: nx.Graph) -> dict:
    """Destroy the top 5 % of edges by betweenness centrality.

    Uses ``nx.global_efficiency`` so the metric stays valid even when
    the network becomes disconnected.
    """
    if G.number_of_edges() < 2:
        return {
            "G_pre": G, "G_post": G,
            "removed": set(), "bottlenecks": set(),
            "eff_pre": 0.0, "eff_post": 0.0,
            "pct_drop": 0.0, "n_components": 1,
            "edges_destroyed": 0,
        }

    eff_pre = nx.global_efficiency(G)

    # Edge betweenness centrality (weight-aware)
    bc = nx.edge_betweenness_centrality(G, weight="w")
    n_remove = max(1, int(math.ceil(len(bc) * _CASCADE_FRACTION)))
    sorted_edges = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    removed = {e for e, _ in sorted_edges[:n_remove]}

    # Build damaged graph
    G_post = G.copy()
    G_post.remove_edges_from(list(removed))

    eff_post = nx.global_efficiency(G_post)
    pct_drop = ((eff_pre - eff_post) / eff_pre * 100) if eff_pre > 0 else 0.0
    n_comp   = nx.number_connected_components(G_post)

    # Identify new traffic bottlenecks in the damaged network
    bottlenecks: set = set()
    if G_post.number_of_edges() > 0:
        new_bc = nx.edge_betweenness_centrality(G_post, weight="w")
        n_bn   = max(1, int(math.ceil(len(new_bc) * _BOTTLENECK_FRACTION)))
        bottlenecks = {e for e, _ in sorted(new_bc.items(),
                                            key=lambda x: x[1],
                                            reverse=True)[:n_bn]}

    return {
        "G_pre": G, "G_post": G_post,
        "removed": removed, "bottlenecks": bottlenecks,
        "eff_pre": eff_pre, "eff_post": eff_post,
        "pct_drop": pct_drop, "n_components": n_comp,
        "edges_destroyed": n_remove,
    }


# =============================================================================
# HTML HELPER  -  glass metric card
# =============================================================================
def _metric_card(label: str, value: str, color: str = "",
                 delta: str = "") -> str:
    delta_html = ""
    if delta:
        delta_color = "#f43f5e" if delta.startswith("-") else "#22c55e"
        delta_html = (f'<div class="glass-delta" '
                      f'style="color:{delta_color}">{delta}</div>')
    return (
        f'<div class="glass-card">'
        f'  <div class="glass-label">{label}</div>'
        f'  <div class="glass-value {color}">{value}</div>'
        f'  {delta_html}'
        f'</div>'
    )


def _gauge_bar(pct: float) -> str:
    """Render a horizontal gauge coloured by resilience percentage."""
    clamped = max(0.0, min(100.0, pct))
    if clamped >= 70:
        colour = "linear-gradient(90deg, #22c55e, #4ade80)"
    elif clamped >= 40:
        colour = "linear-gradient(90deg, #f59e0b, #fbbf24)"
    else:
        colour = "linear-gradient(90deg, #ef4444, #f43f5e)"
    return (
        f'<div class="gauge-outer">'
        f'  <div class="gauge-fill" style="width:{clamped:.1f}%;'
        f'       background:{colour}"></div>'
        f'</div>'
    )


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # -- Header ---------------------------------------------------------------
    st.markdown(
        '<div class="hero-header">'
        '  <p class="hero-title">NetSight</p>'
        '  <p class="hero-subtitle">'
        '    Road Network Extraction  &bull;  Cascade Failure Simulation  '
        '    &bull;  Infrastructure Resilience Analysis'
        '  </p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # -- Sidebar controls -----------------------------------------------------
    with st.sidebar:
        st.markdown("### Controls")

        threshold = st.slider(
            "Detection threshold",
            min_value=0.05, max_value=0.95, value=_DEFAULT_THRESHOLD, step=0.05,
            help="Probability cut-off for the binary road mask.",
        )
        prune_px = st.slider(
            "Min edge length (px)",
            min_value=0, max_value=80, value=8, step=1,
            help="Skeleton spurs shorter than this are pruned.",
        )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### Upload")
        uploaded = st.file_uploader(
            "Satellite tile",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            help="RGB satellite image. Will be resized to 512 x 512 for inference.",
        )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(
            "<small style='color:#475569'>"
            "Powered by ViT-UNet (base_ch=64, depth=4)<br>"
            "Inference: CPU  |  Graph: NetworkX<br>"
            "Resilience metric: Global Efficiency"
            "</small>",
            unsafe_allow_html=True,
        )

    # -- Guard: nothing uploaded yet ------------------------------------------
    if uploaded is None:
        st.info(
            "Upload a satellite tile in the sidebar to begin analysis.",
            icon="\U0001F6F0",
        )
        st.stop()

    # -- Detect file change and reset cascade state ---------------------------
    file_key = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("_file_key") != file_key:
        st.session_state["_file_key"] = file_key
        st.session_state.pop("cascade", None)

    # -- Load + preprocess ----------------------------------------------------
    try:
        pil_img = Image.open(uploaded)
    except Exception:
        # Fallback for exotic GeoTIFF variants
        try:
            import rasterio
            from rasterio.io import MemoryFile
            uploaded.seek(0)
            with MemoryFile(uploaded.read()) as mf, mf.open() as src:
                bands = min(src.count, 3)
                arr = src.read(list(range(1, bands + 1)))
                if bands == 1:
                    arr = np.repeat(arr, 3, axis=0)
                arr = np.transpose(arr, (1, 2, 0))
                pil_img = Image.fromarray(arr.astype(np.uint8))
        except Exception as err:
            st.error(f"Could not read image: {err}")
            st.stop()

    img_array = _preprocess(pil_img)

    # -- Inference ------------------------------------------------------------
    model = _load_model()
    with st.spinner("Running ViT-UNet inference on CPU ..."):
        probs, mask = _run_inference(model, img_array, threshold)

    # -- Graph extraction -----------------------------------------------------
    with st.spinner("Extracting road network graph G(V, E) ..."):
        G, skeleton, nodes = mask_to_graph(mask, prune_min_length=float(prune_px))

    # -- Row 1:  three-panel visualisation ------------------------------------
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        fig1 = _plot_original(img_array)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with col2:
        fig2 = _plot_mask(img_array, probs, mask)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with col3:
        fig3 = _plot_network(G, mask.shape, skeleton)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    # -- Row 2:  network statistics cards -------------------------------------
    road_px    = int(mask.sum())
    skel_px    = int(skeleton.sum())
    n_nodes    = G.number_of_nodes()
    n_edges    = G.number_of_edges()
    glob_eff   = nx.global_efficiency(G) if n_nodes > 1 else 0.0

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    mc = st.columns(5, gap="small")
    cards = [
        ("Nodes |V|",         str(n_nodes),        "cyan"),
        ("Edges |E|",         str(n_edges),         "purple"),
        ("Road Pixels",       f"{road_px:,}",       ""),
        ("Skeleton Pixels",   f"{skel_px:,}",       ""),
        ("Global Efficiency", f"{glob_eff:.4f}",    "green"),
    ]
    for col, (lbl, val, clr) in zip(mc, cards):
        col.markdown(_metric_card(lbl, val, clr), unsafe_allow_html=True)

    # -- Row 3:  CASCADE FAILURE button ---------------------------------------
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if n_edges < 2:
        st.warning(
            "The extracted graph has fewer than 2 edges -- "
            "cascade simulation requires a non-trivial network.",
            icon="\u26A0\uFE0F",
        )
    else:
        if st.button("\U0001F4A5  Simulate Cascade Failure",
                     type="primary", use_container_width=True):
            with st.spinner("Calculating edge betweenness centrality ..."):
                result = _simulate_cascade(G)
            st.session_state["cascade"] = result

    # -- Row 4:  RESILIENCE DASHBOARD (shown after cascade) -------------------
    if "cascade" in st.session_state:
        cr = st.session_state["cascade"]

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-header" style="border-color:rgba(244,63,94,0.25)">'
            '  <p class="hero-title" '
            '     style="background:linear-gradient(135deg,#f43f5e,#f59e0b);'
            '            -webkit-background-clip:text">'
            '    Resilience Index Dashboard</p>'
            '  <p class="hero-subtitle">'
            '    Post-cascade network health after destroying the top 5 %% '
            '    of edges ranked by betweenness centrality'
            '  </p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # -- Resilience gauge -------------------------------------------------
        resilience_pct = max(0.0, 100.0 - cr["pct_drop"])
        st.markdown(
            f"<div style='text-align:center;margin-bottom:0.5rem'>"
            f"  <span style='font-size:2.4rem;font-weight:800;"
            f"  color:{('#22c55e' if resilience_pct >= 70 else '#f59e0b' if resilience_pct >= 40 else '#f43f5e')}'>"
            f"    {resilience_pct:.1f} %"
            f"  </span>"
            f"  <span style='color:#64748b;font-size:0.9rem'>"
            f"    network resilience</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(_gauge_bar(resilience_pct), unsafe_allow_html=True)

        st.markdown("")  # spacer

        # -- Metric cards row -------------------------------------------------
        rc = st.columns(5, gap="small")
        eff_delta = f"-{cr['pct_drop']:.1f} %"
        res_cards = [
            ("Pre-Failure Eff.",  f"{cr['eff_pre']:.4f}",        "green",  ""),
            ("Post-Failure Eff.", f"{cr['eff_post']:.4f}",       "amber",  eff_delta),
            ("Efficiency Drop",   f"{cr['pct_drop']:.1f} %",    "rose",   ""),
            ("Edges Destroyed",   str(cr["edges_destroyed"]),    "rose",   ""),
            ("Components",        str(cr["n_components"]),       "purple", ""),
        ]
        for col, (lbl, val, clr, dlt) in zip(rc, res_cards):
            col.markdown(_metric_card(lbl, val, clr, dlt), unsafe_allow_html=True)

        st.markdown("")  # spacer

        # -- Before / After graph comparison ----------------------------------
        bcol1, bcol2 = st.columns(2, gap="medium")

        with bcol1:
            fig_pre = _plot_network(
                cr["G_pre"], mask.shape, skeleton,
                title="Pre-Failure Network",
            )
            st.pyplot(fig_pre, use_container_width=True)
            plt.close(fig_pre)

        with bcol2:
            # On the pre-graph (which still has all edge paths), mark
            # removed edges as dashed-red and new bottlenecks as amber.
            fig_post = _plot_network(
                cr["G_pre"], mask.shape, skeleton,
                removed_edges=cr["removed"],
                bottleneck_edges=cr["bottlenecks"],
                title="Post-Failure  |  Bottlenecks Highlighted",
            )
            st.pyplot(fig_post, use_container_width=True)
            plt.close(fig_post)


# =============================================================================
if __name__ == "__main__":
    main()
