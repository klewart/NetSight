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
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import streamlit as st
import torch
from PIL import Image
import folium
from streamlit_folium import st_folium

# -- Path injection so we can import project modules --
ROOT = Path(__file__).resolve().parent

try:
    import segmentation_models_pytorch as smp
except ImportError:
    import streamlit as st
    st.error("Please install the required dependency for the new GOD-tier model:\n`pip install segmentation-models-pytorch`")
    st.stop()

from my_agents.src.model import ResNetUNet, RoadSegModel, TestTimeAugmentor, generate_colored_mask, mask_from_graph_criticality
from my_agents.src.mask_to_graph import mask_to_graph
from my_agents.src.network_analysis import (
    compute_node_centrality, compute_edge_criticality, run_node_ablation, simulate_disaster, full_analysis,
)

# -- Constants ----------------------------------------------------------------
_INPUT_SIZE = 512           # Must be divisible by 16 (4x MaxPool2d stride-2)
_DEFAULT_THRESHOLD = 0.50

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
@st.cache_resource(show_spinner="Loading model weights ...")
def _load_model(use_industry: bool = False) -> nn.Module:
    """Load the best available model weights, with fallback to legacy ResNet-UNet."""
    # Priority 1: Industry-grade model (new)
    industry_path = ROOT / "my_agents" / "weights" / "model_weights.pth"
    legacy_path = ROOT / "my_agents" / "weights" / "model_weights_india_osm.pth"

    if use_industry and industry_path.exists():
        try:
            model = RoadSegModel(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=1)
            state = torch.load(str(industry_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            model.eval()
            return model
        except Exception:
            pass  # Fall through to legacy

    # Priority 2: Legacy ResNet-UNet
    if legacy_path.exists():
        model = ResNetUNet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        state = torch.load(str(legacy_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    # Priority 3: Check for any .pth file
    if industry_path.exists():
        model = ResNetUNet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        state = torch.load(str(industry_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    st.error(f"No model weights found. Checked:\n- `{industry_path}`\n- `{legacy_path}`")
    st.stop()


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
def _run_inference(model: nn.Module,
                   img_array: np.ndarray,
                   threshold: float,
                   use_tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (probability_map, binary_mask) from a float32 HWC image."""
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    # Apply ImageNet normalization to match training data
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    if use_tta:
        tta = TestTimeAugmentor(model, device="cpu")
        probs = tta.predict(tensor).squeeze().cpu().numpy()
    else:
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


def _plot_colored_mask(colored_mask: np.ndarray):
    """Plot the color-coded road mask on black background."""
    fig, ax = _dark_fig()
    ax.imshow(colored_mask)
    ax.set_title("Color-Coded Road Segmentation", color="white", fontsize=13,
                 fontweight="bold", pad=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='#555', label='Normal Road'),
        Patch(facecolor='#ffff00', edgecolor='#555', label='Critical Road'),
        Patch(facecolor='#ff0000', edgecolor='#555', label='Destroyed Road'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              facecolor='#111827', edgecolor='#1e293b', labelcolor='white')
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
                  bg_img: np.ndarray | None = None,
                  removed_edges: set | None = None,
                  bottleneck_edges: set | None = None,
                  title: str = "Road Network  G(V, E)"):
    """Render the road graph.  Optionally mark removed & bottleneck edges."""
    fig, ax = _dark_fig((7, 7))
    H, W = shape
    removed_f    = _normalise_edge_set(removed_edges)
    bottleneck_f = _normalise_edge_set(bottleneck_edges)

    # Darkened satellite image backdrop
    if bg_img is not None:
        ax.imshow(bg_img, alpha=0.35, origin="upper")

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
                    linestyle=(0, (5, 4)), zorder=3,
                    path_effects=[pe.SimpleLineShadow(shadow_color=_C_REMOVED, alpha=0.4, linewidth=6), pe.Normal()])
        elif is_bottle:
            ax.plot(cols, rows, color=_C_BOTTLENCK, lw=3.2, alpha=0.92, zorder=4,
                    path_effects=[pe.SimpleLineShadow(shadow_color=_C_BOTTLENCK, alpha=0.5, linewidth=8), pe.Normal()])
        else:
            ax.plot(cols, rows, color=_C_EDGE, lw=1.6, alpha=0.7, zorder=2,
                    path_effects=[pe.SimpleLineShadow(shadow_color=_C_EDGE, alpha=0.3, linewidth=4), pe.Normal()])

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


def _render_interactive_map(G: nx.Graph, shape: tuple[int, int], bg_img: np.ndarray | None = None,
                            removed_edges: set | None = None, bottleneck_edges: set | None = None,
                            removed_nodes: set | None = None):
    """Render the road graph and Heatmap Overlay using Folium."""
    import base64
    import io
    
    H, W = shape
    # Map bounds for image overlay: map pixels to a pseudo coordinate system
    bounds = [[-H, 0], [0, W]]
    
    m = folium.Map(location=[-H/2, W/2], zoom_start=1, crs='Simple', max_bounds=True)
    m.fit_bounds(bounds)

    if bg_img is not None:
        img = Image.fromarray((bg_img * 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_str}",
            bounds=bounds,
            opacity=0.35,
            interactive=False,
            zindex=1
        ).add_to(m)

    if G.number_of_nodes() == 0:
        return m

    edge_criticality = compute_edge_criticality(G)
    cmap = matplotlib.colormaps['YlOrRd']
    
    removed_f = _normalise_edge_set(removed_edges)
    bottleneck_f = _normalise_edge_set(bottleneck_edges)
    removed_n = set(removed_nodes) if removed_nodes else set()

    # Draw edges
    for u, v, data in G.edges(data=True):
        is_removed = _edge_in(u, v, removed_f)
        is_bottle = _edge_in(u, v, bottleneck_f)
        
        path = data.get("path", [G.nodes[u]["pos"], G.nodes[v]["pos"]])
        line_coords = [[-float(r), float(c)] for r, c in path]
        
        if is_removed:
            folium.PolyLine(line_coords, color="#f43f5e", weight=4, dash_array="10", opacity=0.8).add_to(m)
        elif is_bottle:
            folium.PolyLine(line_coords, color="#f59e0b", weight=5, opacity=0.9).add_to(m)
        else:
            crit = edge_criticality.get(frozenset((u, v)), 0.0)
            rgb = cmap(crit)[:3]
            hex_color = '#{:02x}{:02x}{:02x}'.format(*(int(c * 255) for c in rgb))
            weight = 3 + (crit * 4)
            folium.PolyLine(line_coords, color=hex_color, weight=weight, opacity=0.7).add_to(m)

    # Draw nodes
    for nid, data in G.nodes(data=True):
        if nid in removed_n:
            continue
        r, c = data["pos"]
        coord = [-float(r), float(c)]
        
        is_junc = data.get("kind") == "junction"
        color = "#22c55e" if is_junc else "#a855f7"
        
        folium.CircleMarker(
            location=coord,
            radius=4 if is_junc else 3,
            color="white",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=f"Node {nid} (Click to Disable)",
        ).add_to(m)

    return m

def _generate_animated_network_gif(G: nx.Graph, shape: tuple[int, int], bg_img: np.ndarray, 
                                   removed_edges: set, bottleneck_edges: set, out_path: str):
    """Generate a looping animated GIF showing pulsing destroyed roads and bottlenecks."""
    fig, ax = _dark_fig((7, 7))
    H, W = shape
    removed_f = _normalise_edge_set(removed_edges)
    bottleneck_f = _normalise_edge_set(bottleneck_edges)

    ax.imshow(bg_img, alpha=0.25, origin="upper")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title("Post-Failure | Critical Simulation", color="white", fontsize=13, fontweight="bold", pad=10)

    # Static standard edges
    for u, v, data in G.edges(data=True):
        if not _edge_in(u, v, removed_f) and not _edge_in(u, v, bottleneck_f):
            path = data.get("path", [G.nodes[u]["pos"], G.nodes[v]["pos"]])
            cols, rows = [int(p[1]) for p in path], [int(p[0]) for p in path]
            ax.plot(cols, rows, color=_C_EDGE, lw=1.4, alpha=0.4, zorder=2)

    # Animated edges
    rem_lines, bot_lines = [], []
    for u, v, data in G.edges(data=True):
        path = data.get("path", [G.nodes[u]["pos"], G.nodes[v]["pos"]])
        cols, rows = [int(p[1]) for p in path], [int(p[0]) for p in path]
        if _edge_in(u, v, removed_f):
            l, = ax.plot(cols, rows, color=_C_REMOVED, lw=3.0, linestyle=(0, (5, 4)), zorder=4,
                         path_effects=[pe.SimpleLineShadow(shadow_color=_C_REMOVED, alpha=0.6, linewidth=8), pe.Normal()])
            rem_lines.append(l)
        elif _edge_in(u, v, bottleneck_f):
            l, = ax.plot(cols, rows, color=_C_BOTTLENCK, lw=3.5, zorder=5,
                         path_effects=[pe.SimpleLineShadow(shadow_color=_C_BOTTLENCK, alpha=0.7, linewidth=10), pe.Normal()])
            bot_lines.append(l)

    def update(frame):
        # Pulse between 0.2 and 1.0 based on frame
        alpha_rem = 0.4 + 0.6 * np.sin(frame * np.pi / 5) ** 2
        alpha_bot = 0.5 + 0.5 * np.cos(frame * np.pi / 5) ** 2
        for l in rem_lines: l.set_alpha(alpha_rem)
        for l in bot_lines: l.set_alpha(alpha_bot)
        return rem_lines + bot_lines

    ani = FuncAnimation(fig, update, frames=10, blit=True)
    ani.save(out_path, writer='pillow', fps=8)
    plt.close(fig)


# =============================================================================
# CASCADE FAILURE SIMULATION  (Powered by network_analysis.py engine)
# =============================================================================


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
        use_tta = st.checkbox(
            "Enable TTA (Test-Time Augmentation)",
            value=False,
            help="Runs 4x inference with flips and averages predictions. Slower but more accurate.",
        )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### Upload")
        uploaded = st.file_uploader(
            "Satellite tile",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            help="RGB satellite image. Will be resized to 512 x 512 for inference.",
        )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### Emergency Simulation")

        sim_mode = st.radio("Failure Scenario", ["Analytical (Betweenness)", "Disaster (Blast Radius)"],
                            help="How should roads be destroyed during simulation?")

        blast_radius = 0.0
        if sim_mode == "Disaster (Blast Radius)":
            blast_radius = st.slider(
                "Blast Radius (pixels)",
                min_value=10, max_value=250, value=80, step=10,
                help="Simulates a flood or blast at the image center, wiping out all roads in the radius."
            )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(
            "<small style='color:#475569'>"
            "Powered by Industry-Grade Segmentation<br>"
            "UNet++ / EfficientNet-B4 Encoder<br>"
            "Inference: CPU + TTA  |  Graph: NetworkX<br>"
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
    model = _load_model(use_industry=True)
    tta_label = "TTA" if use_tta else "standard"
    with st.spinner(f"Running {tta_label} inference on CPU ..."):
        probs, mask = _run_inference(model, img_array, threshold, use_tta=use_tta)

    # -- Graph extraction -----------------------------------------------------
    with st.spinner("Extracting road network graph G(V, E) ..."):
        G, skeleton, nodes = mask_to_graph(mask, prune_min_length=float(prune_px))

    # -- Generate color-coded segmentation mask on black background -----------
    criticality_map = mask_from_graph_criticality(mask, G, mask.shape)

    # Check for destroyed roads from disaster simulation
    destroyed_mask_overlay = None
    cr = st.session_state.get("cascade")
    if cr is not None and hasattr(cr, 'removed_edges') and cr.removed_edges:
        destroyed_mask_overlay = np.zeros_like(mask, dtype=np.uint8)
        for u, v in cr.removed_edges:
            path = cr.G_pre[u][v].get('path', []) if cr.G_pre.has_edge(u, v) else []
            for r, c in path:
                r, c = int(r), int(c)
                if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
                    destroyed_mask_overlay[r, c] = 1

    colored_mask = generate_colored_mask(mask, criticality_map, destroyed_mask_overlay)

    # -- Row 1:  four-panel visualisation ------------------------------------
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    col1, col2, col2b, col3 = st.columns([1, 1, 1, 2], gap="medium")

    with col1:
        fig1 = _plot_original(img_array)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with col2:
        fig2 = _plot_mask(img_array, probs, mask)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with col2b:
        fig2b = _plot_colored_mask(colored_mask)
        st.pyplot(fig2b, use_container_width=True)
        plt.close(fig2b)

        # Download button for the color-coded mask
        import io
        mask_pil = Image.fromarray(colored_mask)
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        st.download_button(
            label="📥 Download Mask",
            data=buf.getvalue(),
            file_name="netsight_road_mask.png",
            mime="image/png",
            use_container_width=True,
        )

    with col3:
        st.markdown("**Phase IV Interactive Dashboard** (Click nodes to disable)")
        
        # Check if we have an active disaster state to render
        cr = st.session_state.get("cascade")
        if cr is not None:
            m = _render_interactive_map(
                cr.G_post, mask.shape, img_array, 
                removed_edges=cr.removed_edges, 
                bottleneck_edges=cr.bottleneck_edges,
                removed_nodes=cr.removed_nodes
            )
        else:
            m = _render_interactive_map(G, mask.shape, img_array)
            
        st_data = st_folium(m, use_container_width=True, height=450, key="interactive_map", returned_objects=["last_object_clicked"])
        
        # Handle Node Click Event for Simulation Toggle
        if st_data and st_data.get("last_object_clicked"):
            click = st_data["last_object_clicked"]
            lat, lng = click["lat"], click["lng"]
            
            # Prevent infinite reruns by checking if we already processed this click
            last_click = st.session_state.get("last_processed_click")
            if last_click != (lat, lng):
                st.session_state["last_processed_click"] = (lat, lng)
                
                # Find the nearest node to the click
                best_nid = None
                min_dist = float("inf")
                for nid, data in G.nodes(data=True):
                    r, c = data["pos"]
                    n_lat, n_lng = -float(r), float(c)
                    dist = (lat - n_lat)**2 + (lng - n_lng)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_nid = nid
                        
                # If click is within reasonable pixel distance (e.g., 20^2 = 400)
                if best_nid is not None and min_dist < 400:
                    with st.spinner(f"Simulating targeted failure at Node {best_nid}..."):
                        pos = G.nodes[best_nid]["pos"]
                        disaster = simulate_disaster(
                            G,
                            epicenter=(float(pos[0]), float(pos[1])),
                            blast_radius_px=5.0,  # Tiny radius to just destroy the clicked node
                            fallback_fraction=0.0
                        )
                        st.session_state["cascade"] = disaster
                    st.rerun()

    # -- Row 2:  network statistics + Gatekeeper Nodes -------------------------
    road_px    = int(mask.sum())
    skel_px    = int(skeleton.sum())
    n_nodes    = G.number_of_nodes()
    n_edges    = G.number_of_edges()
    glob_eff   = nx.global_efficiency(G) if n_nodes > 1 else 0.0
    n_healed   = G.graph.get('n_healed_edges', 0)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    mc = st.columns(6, gap="small")
    cards = [
        ("Nodes |V|",         str(n_nodes),        "cyan"),
        ("Edges |E|",         str(n_edges),         "purple"),
        ("Healed Edges",      str(n_healed),        "green" if n_healed > 0 else ""),
        ("Road Pixels",       f"{road_px:,}",       ""),
        ("Skeleton Pixels",   f"{skel_px:,}",       ""),
        ("Global Efficiency", f"{glob_eff:.4f}",    "green"),
    ]
    for col, (lbl, val, clr) in zip(mc, cards):
        col.markdown(_metric_card(lbl, val, clr), unsafe_allow_html=True)

    # -- Row 2b: Gatekeeper Nodes table ----------------------------------------
    if n_nodes >= 2:
        gatekeepers = compute_node_centrality(G, top_k=min(10, n_nodes))
        if gatekeepers:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown(
                '<div class="hero-header" style="border-color:rgba(168,85,247,0.25)">'
                '  <p class="hero-title" '
                '     style="background:linear-gradient(135deg,#a855f7,#00d4ff);'
                '            -webkit-background-clip:text">'
                '    Gatekeeper Nodes</p>'
                '  <p class="hero-subtitle">'
                '    Critical intersections ranked by Betweenness Centrality &mdash; '
                '    single points of failure in the road network'
                '  </p>'
                '</div>',
                unsafe_allow_html=True,
            )
            gk_data = {
                "Rank": [gk.rank for gk in gatekeepers],
                "Node ID": [gk.node_id for gk in gatekeepers],
                "Position (r, c)": [f"({gk.pos[0]}, {gk.pos[1]})" for gk in gatekeepers],
                "Type": [gk.kind.title() for gk in gatekeepers],
                "Degree": [gk.degree for gk in gatekeepers],
                "Betweenness": [f"{gk.betweenness:.6f}" for gk in gatekeepers],
            }
            st.dataframe(gk_data, use_container_width=True, hide_index=True)

    # -- Row 3:  STRESS TEST buttons ------------------------------------------
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if n_nodes < 3:
        st.warning(
            "The extracted graph has fewer than 3 nodes -- "
            "stress testing requires a non-trivial network.",
            icon="\u26A0\uFE0F",
        )
    else:
        stress_col1, stress_col2 = st.columns(2, gap="medium")

        with stress_col1:
            if st.button("\U0001F4A5  Simulate Disaster",
                         type="primary", use_container_width=True):
                with st.spinner("Running node-centric disaster simulation..."):
                    disaster = simulate_disaster(
                        G,
                        epicenter=(_INPUT_SIZE // 2, _INPUT_SIZE // 2),
                        blast_radius_px=blast_radius if blast_radius > 0 else 0.0,
                        fallback_fraction=0.05,
                    )
                st.session_state["cascade"] = disaster

        with stress_col2:
            if st.button("\U0001F9EA  Run Node Ablation Study",
                         type="secondary", use_container_width=True):
                with st.spinner("Running systematic node ablation (adaptive re-ranking)..."):
                    ablation = run_node_ablation(G, fraction=0.30)
                st.session_state["ablation"] = ablation

    # -- Row 4:  RESILIENCE DASHBOARD (shown after disaster) -------------------
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
            '    Post-disaster network health &mdash; node-centric simulation '
            '    with formal Resilience Index R = E<sub>post</sub> / E<sub>pre</sub>'
            '  </p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # -- Resilience gauge -------------------------------------------------
        resilience_pct = cr.resilience_index * 100.0
        st.markdown(
            f"<div style='text-align:center;margin-bottom:0.5rem'>"
            f"  <span style='font-size:2.4rem;font-weight:800;"
            f"  color:{('#22c55e' if resilience_pct >= 70 else '#f59e0b' if resilience_pct >= 40 else '#f43f5e')}'>"
            f"    R = {cr.resilience_index:.4f}"
            f"  </span>"
            f"  <span style='color:#64748b;font-size:0.9rem'>"
            f"    ({resilience_pct:.1f}% network resilience)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(_gauge_bar(resilience_pct), unsafe_allow_html=True)

        st.markdown("")  # spacer

        # -- Metric cards row -------------------------------------------------
        rc1 = st.columns(4, gap="small")
        rc2 = st.columns(4, gap="small")

        eff_delta = f"-{cr.pct_drop:.1f} %"
        time_pen = f"+{cr.time_penalty_pct:.1f} %" if cr.time_penalty_pct < 999.0 else "INF"

        res_cards1 = [
            ("Pre-Failure Eff.",  f"{cr.eff_pre:.4f}",        "green",  ""),
            ("Post-Failure Eff.", f"{cr.eff_post:.4f}",       "amber",  eff_delta),
            ("Efficiency Drop",   f"{cr.pct_drop:.1f} %",    "rose",   ""),
            ("Evac. Time Penalty", time_pen,                  "rose",   ""),
        ]
        res_cards2 = [
            ("Nodes Destroyed",   str(cr.nodes_destroyed),    "rose",   ""),
            ("Edges Destroyed",   str(cr.edges_destroyed),    "rose",   ""),
            ("Isolated Nodes",    str(cr.isolated_nodes),     "rose" if cr.isolated_nodes > 0 else "green", ""),
            ("Total Components",  str(cr.n_components),       "purple", ""),
        ]

        for col, (lbl, val, clr, dlt) in zip(rc1, res_cards1):
            col.markdown(_metric_card(lbl, val, clr, dlt), unsafe_allow_html=True)
        for col, (lbl, val, clr, dlt) in zip(rc2, res_cards2):
            col.markdown(_metric_card(lbl, val, clr, dlt), unsafe_allow_html=True)

        st.markdown("")  # spacer

        # -- Before / After graph comparison ----------------------------------
        bcol1, bcol2 = st.columns(2, gap="medium")

        with bcol1:
            fig_pre = _plot_network(
                cr.G_pre, mask.shape, img_array,
                title="Pre-Failure Network",
            )
            st.pyplot(fig_pre, use_container_width=True)
            plt.close(fig_pre)

        with bcol2:
            import tempfile
            import base64
            with st.spinner("Generating animation..."):
                with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
                    _generate_animated_network_gif(
                        cr.G_pre, mask.shape, img_array,
                        cr.removed_edges, cr.bottleneck_edges, tmp.name
                    )
                    with open(tmp.name, "rb") as f:
                        gif_data = f.read()

            b64 = base64.b64encode(gif_data).decode()
            st.markdown(
                f'<img src="data:image/gif;base64,{b64}" width="100%" '
                f'style="border-radius:10px; border:2px solid #1e293b">',
                unsafe_allow_html=True
            )

    # -- Row 5:  ABLATION STUDY (shown after ablation button) -----------------
    if "ablation" in st.session_state:
        ab = st.session_state["ablation"]

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-header" style="border-color:rgba(0,212,255,0.25)">'
            '  <p class="hero-title" '
            '     style="background:linear-gradient(135deg,#00d4ff,#22c55e);'
            '            -webkit-background-clip:text">'
            '    Node Ablation Study</p>'
            '  <p class="hero-subtitle">'
            '    Systematic removal of critical nodes with adaptive re-ranking &mdash; '
            '    degradation curve shows network fragility trajectory'
            '  </p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # -- Ablation summary cards -------------------------------------------
        ab_cols = st.columns(4, gap="small")
        ab_cards = [
            ("Nodes Removed",     str(ab.total_nodes_removed),           "rose",   ""),
            ("Baseline Eff.",     f"{ab.baseline_efficiency:.4f}",       "green",  ""),
            ("Final Eff.",        f"{ab.final_efficiency:.4f}",          "amber",  ""),
            ("Final R-Index",     f"{ab.final_resilience_index:.4f}",    "rose" if ab.final_resilience_index < 0.5 else "green", ""),
        ]
        for col, (lbl, val, clr, dlt) in zip(ab_cols, ab_cards):
            col.markdown(_metric_card(lbl, val, clr, dlt), unsafe_allow_html=True)

        st.markdown("")  # spacer

        # -- Degradation curve chart ------------------------------------------
        steps_x = [s.step for s in ab.steps]
        eff_y = [s.global_efficiency for s in ab.steps]
        ri_y = [s.resilience_index for s in ab.steps]
        comp_y = [s.n_components for s in ab.steps]

        fig_deg, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=_FIG_BG)

        # Left: Efficiency + Resilience Index
        ax1.set_facecolor(_AX_BG)
        ax1.plot(steps_x, eff_y, color='#00d4ff', linewidth=2.5, marker='o',
                 markersize=5, label='Global Efficiency', zorder=3)
        ax1.plot(steps_x, ri_y, color='#f59e0b', linewidth=2.0, marker='s',
                 markersize=4, linestyle='--', label='Resilience Index', zorder=3)
        ax1.fill_between(steps_x, eff_y, alpha=0.15, color='#00d4ff')
        ax1.set_xlabel('Ablation Step (nodes removed)', color='#94a3b8', fontsize=10)
        ax1.set_ylabel('Value', color='#94a3b8', fontsize=10)
        ax1.set_title('Degradation Curve', color='white', fontsize=13, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', fontsize=8, facecolor='#111827',
                   edgecolor='#1e293b', labelcolor='white')
        ax1.tick_params(colors='#64748b')
        ax1.grid(True, alpha=0.1, color='#475569')
        for sp in ax1.spines.values():
            sp.set_color('#1e293b')

        # Right: Component fragmentation
        ax2.set_facecolor(_AX_BG)
        ax2.bar(steps_x, comp_y, color='#a855f7', alpha=0.8, edgecolor='#c084fc', linewidth=0.5)
        ax2.set_xlabel('Ablation Step', color='#94a3b8', fontsize=10)
        ax2.set_ylabel('Connected Components', color='#94a3b8', fontsize=10)
        ax2.set_title('Network Fragmentation', color='white', fontsize=13, fontweight='bold', pad=10)
        ax2.tick_params(colors='#64748b')
        ax2.grid(True, alpha=0.1, color='#475569', axis='y')
        for sp in ax2.spines.values():
            sp.set_color('#1e293b')

        plt.tight_layout()
        st.pyplot(fig_deg, use_container_width=True)
        plt.close(fig_deg)

        # -- Ablation step detail table ---------------------------------------
        st.markdown("")  # spacer
        with st.expander("Ablation Step Details", expanded=False):
            abl_data = {
                "Step": [s.step for s in ab.steps],
                "Removed Node": [str(s.removed_node_id) if s.removed_node_id is not None else "---" for s in ab.steps],
                "Position": [str(s.removed_node_pos) if s.removed_node_pos else "---" for s in ab.steps],
                "Efficiency": [f"{s.global_efficiency:.4f}" for s in ab.steps],
                "R-Index": [f"{s.resilience_index:.4f}" for s in ab.steps],
                "Components": [s.n_components for s in ab.steps],
                "Isolated": [s.isolated_nodes for s in ab.steps],
            }
            st.dataframe(abl_data, use_container_width=True, hide_index=True)


# =============================================================================
if __name__ == "__main__":
    main()
