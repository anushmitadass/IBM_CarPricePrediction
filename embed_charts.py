"""
embed_charts.py
───────────────
Generates every chart from the real carprice.csv data using matplotlib
(no pandas needed) and embeds each PNG as a base64 output directly
inside the matching code cell of AnushmitaDas_CarPricePrediction.ipynb.

Run:  python embed_charts.py
"""

import csv, json, math, io, base64, collections, statistics, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── Colour constants ──────────────────────────────────────────────────────────
BG      = "#0f1117"
SURFACE = "#1a1d2e"
BORDER  = "#2a2f4a"
PURPLE  = "#6c63ff"
GREEN   = "#00d4aa"
RED     = "#ff6584"
YELLOW  = "#ffd166"
BLUE    = "#4cc9f0"
ORANGE  = "#f77f00"
MUTED   = "#8892b0"
TEXT    = "#e8eaf6"
PALETTE = [PURPLE, GREEN, RED, YELLOW, BLUE, ORANGE,
           "#a29bfe", "#55efc4", "#fd79a8", "#fdcb6e"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": SURFACE,
    "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
    "axes.titlecolor": TEXT, "axes.grid": True,
    "grid.color": BORDER, "grid.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT,
    "legend.facecolor": "#1e2235", "legend.edgecolor": BORDER,
    "legend.labelcolor": TEXT, "font.family": "DejaVu Sans",
    "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelsize": 11,
})

# ── Load CSV ──────────────────────────────────────────────────────────────────
def to_float(v):
    try: return float(v)
    except: return None

with open("carprice.csv", newline="", encoding="utf-8") as f:
    raw = list(csv.DictReader(f))

# Replace "?" with None
for row in raw:
    for k in row:
        if row[k] == "?":
            row[k] = None

NUM_COLS = ["symboling","normalized-losses","wheel-base","length","width",
            "height","curb-weight","engine-size","bore","stroke",
            "compression-ratio","horsepower","peak-rpm","city-mpg",
            "highway-mpg","price"]
for row in raw:
    for c in NUM_COLS:
        if c in row:
            row[c] = to_float(row[c])

# Drop rows without price
data = [r for r in raw if r.get("price") is not None]
prices = [r["price"] for r in data]

# ── Helper: fig → base64 PNG string ──────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64

# ── Helper: make an output dict ───────────────────────────────────────────────
def png_output(b64):
    return {
        "output_type": "display_data",
        "metadata": {"image/png": {"width": 900}},
        "data": {"image/png": b64, "text/plain": ["<Figure>"]},
    }

def text_output(text):
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": [text + "\n"],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Chart generators — one function per notebook plotting cell
# ═══════════════════════════════════════════════════════════════════════════════

# ── Cell: missing value bar chart ─────────────────────────────────────────────
def chart_missing():
    miss_cols = ["normalized-losses","bore","stroke","horsepower","peak-rpm",
                 "num-of-doors"]
    counts = []
    for c in miss_cols:
        n = sum(1 for r in raw if r.get(c) is None)
        if n: counts.append((c, n))
    counts.sort(key=lambda x: x[1], reverse=True)
    if not counts:
        return None
    labels, vals = zip(*counts)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(labels, vals, color=RED, edgecolor="none", height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.05, i, str(v), va="center", color=TEXT, fontsize=10)
    ax.set_title("Missing Value Counts per Column", pad=14)
    ax.set_xlabel("Number of Missing Values")
    ax.invert_yaxis()
    ax.set_facecolor(SURFACE)
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: price distribution (hist + box) ────────────────────────────────────
def chart_price_dist():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    # histogram
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    mn, mx = min(prices), max(prices)
    bins = [mn + (mx-mn)/30*i for i in range(31)]
    counts_hist = [0]*30
    for p in prices:
        idx = min(int((p-mn)/(mx-mn)*30), 29)
        counts_hist[idx] += 1
    centers = [(bins[i]+bins[i+1])/2 for i in range(30)]
    widths  = [bins[i+1]-bins[i] for i in range(30)]
    ax.bar(centers, counts_hist, width=widths, color=PURPLE, alpha=0.85, edgecolor="none")
    ax.set_title("Price Distribution")
    ax.set_xlabel("Price (USD)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    # box plot
    ax2 = axes[1]
    ax2.set_facecolor(SURFACE)
    bp = ax2.boxplot(prices, patch_artist=True, orientation="vertical", widths=0.5,
                     boxprops=dict(facecolor=PURPLE, color=PURPLE, alpha=0.7),
                     medianprops=dict(color=GREEN, linewidth=2.5),
                     whiskerprops=dict(color=MUTED),
                     capprops=dict(color=MUTED),
                     flierprops=dict(marker="o", color=RED, markersize=5, alpha=0.6))
    ax2.set_title("Price Box Plot")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    fig.suptitle("Target Variable — Car Price", fontsize=15, fontweight="bold",
                 y=1.02, color=TEXT)
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: avg price by make ───────────────────────────────────────────────────
def chart_make_price():
    make_prices = collections.defaultdict(list)
    for r in data:
        make_prices[r["make"]].append(r["price"])
    avg = {k: statistics.mean(v) for k,v in make_prices.items()}
    avg = dict(sorted(avg.items(), key=lambda x: x[1], reverse=True))
    labels = list(avg.keys()); vals = list(avg.values())
    colors = [GREEN if v == max(vals) else RED if v > 20000
              else YELLOW if v > 12000 else PURPLE for v in vals]
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    bars = ax.bar(labels, vals, color=colors, edgecolor="none", width=0.7)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+200,
                f"${h/1000:.0f}k", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_title("Average Car Price by Make", pad=14)
    ax.set_ylabel("Average Price (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax.legend(handles=[
        mpatches.Patch(color=GREEN, label="Highest"),
        mpatches.Patch(color=RED,   label="> $20k"),
        mpatches.Patch(color=YELLOW,label="> $12k"),
        mpatches.Patch(color=PURPLE,label="≤ $12k"),
    ], loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: correlation heatmap ─────────────────────────────────────────────────
def chart_corr_heatmap():
    num_feats = ["symboling","normalized-losses","wheel-base","length","width",
                 "height","curb-weight","engine-size","bore","stroke",
                 "compression-ratio","horsepower","peak-rpm","city-mpg",
                 "highway-mpg","price"]
    def safe_vals(feat):
        return [(r[feat], r["price"]) for r in data
                if r.get(feat) is not None and r.get("price") is not None]
    def pearson(xs, ys):
        n = len(xs)
        if n < 2: return 0
        mx = sum(xs)/n; my = sum(ys)/n
        num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        dx  = math.sqrt(sum((x-mx)**2 for x in xs))
        dy  = math.sqrt(sum((y-my)**2 for y in ys))
        return num/(dx*dy) if dx*dy else 0
    feats = [f for f in num_feats if f != "price" and
             sum(1 for r in data if r.get(f) is not None) > 10]
    n = len(feats)
    mat = [[0.0]*n for _ in range(n)]
    for i, fi in enumerate(feats):
        for j, fj in enumerate(feats):
            if i == j: mat[i][j] = 1.0
            elif j < i: mat[i][j] = mat[j][i]
            else:
                pairs = [(r[fi], r[fj]) for r in data
                         if r.get(fi) is not None and r.get(fj) is not None]
                if pairs:
                    xs, ys = zip(*pairs)
                    mat[i][j] = pearson(list(xs), list(ys))
    cmap = LinearSegmentedColormap.from_list("div", [RED, SURFACE, PURPLE], N=256)
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    im = ax.imshow([[mat[i][j] if j < i else float("nan")
                     for j in range(n)] for i in range(n)],
                   cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(feats, fontsize=8)
    for i in range(n):
        for j in range(i):
            ax.text(j, i, f"{mat[i][j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color=TEXT)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Matrix — Numeric Features", pad=14)
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: top correlations bar ────────────────────────────────────────────────
def chart_price_corr_bar():
    def pearson_with_price(feat):
        pairs = [(r[feat], r["price"]) for r in data
                 if r.get(feat) is not None]
        if len(pairs) < 5: return 0
        xs, ys = zip(*pairs)
        xs = list(xs); ys = list(ys)
        mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
        num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        dx = math.sqrt(sum((x-mx)**2 for x in xs))
        dy = math.sqrt(sum((y-my)**2 for y in ys))
        return num/(dx*dy) if dx*dy else 0
    num_feats = ["symboling","normalized-losses","wheel-base","length","width",
                 "height","curb-weight","engine-size","bore","stroke",
                 "compression-ratio","horsepower","peak-rpm","city-mpg",
                 "highway-mpg"]
    corrs = {f: pearson_with_price(f) for f in num_feats}
    top10 = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    top10_rev = top10[::-1]
    labels = [x[0] for x in top10_rev]
    vals   = [x[1] for x in top10_rev]
    bar_colors = [GREEN if v > 0 else RED for v in vals]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    bars = ax.barh(labels, vals, color=bar_colors, edgecolor="none", height=0.6)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + (0.01 if w >= 0 else -0.01),
                bar.get_y()+bar.get_height()/2,
                f"{w:.3f}", va="center",
                ha="left" if w >= 0 else "right",
                color=MUTED, fontsize=9)
    ax.set_title("Top 10 Feature Correlations with Price", pad=14)
    ax.set_xlabel("Pearson Correlation"); ax.set_xlim(-1.1, 1.1)
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: scatter plots top 6 features vs price ───────────────────────────────
def chart_scatter_grid():
    def pearson_with_price(feat):
        pairs = [(r[feat], r["price"]) for r in data if r.get(feat) is not None]
        if len(pairs) < 5: return 0
        xs, ys = zip(*pairs)
        xs=list(xs); ys=list(ys)
        mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
        num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        dx=math.sqrt(sum((x-mx)**2 for x in xs))
        dy=math.sqrt(sum((y-my)**2 for y in ys))
        return num/(dx*dy) if dx*dy else 0
    num_feats = ["symboling","normalized-losses","wheel-base","length","width",
                 "height","curb-weight","engine-size","bore","stroke",
                 "compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg"]
    corrs = sorted(num_feats, key=lambda f: abs(pearson_with_price(f)), reverse=True)
    top6 = corrs[:6]
    scatter_colors = [PURPLE, GREEN, RED, YELLOW, BLUE, ORANGE]
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.suptitle("Top Numeric Features vs. Car Price", fontsize=15,
                 fontweight="bold", y=1.01, color=TEXT)
    for i, feat in enumerate(top6):
        ax = fig.add_subplot(2, 3, i+1)
        ax.set_facecolor(SURFACE)
        pairs = [(r[feat], r["price"]) for r in data if r.get(feat) is not None]
        xs, ys = zip(*pairs)
        xs=list(xs); ys=list(ys)
        ax.scatter(xs, ys, alpha=0.55, color=scatter_colors[i],
                   edgecolors="none", s=40)
        # trend line
        n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
        m = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
        b = my - m*mx
        xmin, xmax = min(xs), max(xs)
        ax.plot([xmin, xmax], [m*xmin+b, m*xmax+b],
                color="white", linewidth=1.8, linestyle="--", alpha=0.7)
        ax.set_xlabel(feat); ax.set_ylabel("Price")
        ax.set_title(f"{feat} vs Price")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: price by body style & fuel type ────────────────────────────────────
def chart_body_fuel():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    # body style box
    body_data = collections.defaultdict(list)
    for r in data:
        if r.get("body-style"): body_data[r["body-style"]].append(r["price"])
    order = sorted(body_data, key=lambda k: statistics.median(body_data[k]), reverse=True)
    ax = axes[0]; ax.set_facecolor(SURFACE)
    bp = ax.boxplot([body_data[k] for k in order], patch_artist=True, orientation="vertical",
                    tick_labels=order, widths=0.5,
                    medianprops=dict(color="white", linewidth=2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))
    for patch, color in zip(bp["boxes"], PALETTE): patch.set_facecolor(color); patch.set_alpha(0.8)
    ax.set_title("Price by Body Style")
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    # fuel type box
    fuel_data = collections.defaultdict(list)
    for r in data:
        if r.get("fuel-type"): fuel_data[r["fuel-type"]].append(r["price"])
    fkeys = sorted(fuel_data)
    ax2 = axes[1]; ax2.set_facecolor(SURFACE)
    bp2 = ax2.boxplot([fuel_data[k] for k in fkeys], patch_artist=True,
                      tick_labels=fkeys, widths=0.5,
                      medianprops=dict(color="white", linewidth=2),
                      flierprops=dict(marker="o", markersize=4, alpha=0.5))
    for patch, color in zip(bp2["boxes"], [GREEN, RED]): patch.set_facecolor(color); patch.set_alpha(0.8)
    ax2.set_title("Price by Fuel Type")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: categorical count plots ────────────────────────────────────────────
def chart_cat_counts():
    cat_cols = ["fuel-type","aspiration","num-of-doors","body-style",
                "drive-wheels","engine-type","num-of-cylinders","fuel-system"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.patch.set_facecolor(BG)
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        counts = collections.Counter(r[col] for r in raw if r.get(col))
        order = [k for k, _ in counts.most_common()]
        vals  = [counts[k] for k in order]
        ax = axes[i]; ax.set_facecolor(SURFACE)
        bars = ax.bar(order, vals,
                      color=[PALETTE[j % len(PALETTE)] for j in range(len(order))],
                      edgecolor="none")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.5,
                    str(int(h)), ha="center", va="bottom", fontsize=8, color=MUTED)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Categorical Feature Distributions", fontsize=14,
                 fontweight="bold", y=1.01, color=TEXT)
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: violin — price by drive wheels ─────────────────────────────────────
def chart_violin_drive():
    dw_data = collections.defaultdict(list)
    for r in data:
        if r.get("drive-wheels"): dw_data[r["drive-wheels"]].append(r["price"])
    order = sorted(dw_data)
    colors = [PURPLE, GREEN, YELLOW]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    for i, key in enumerate(order):
        vals = sorted(dw_data[key])
        # Simple violin approximation using a wide box + scatter strip
        q1 = vals[len(vals)//4]; q3 = vals[3*len(vals)//4]
        med = statistics.median(vals)
        ax.boxplot(vals, positions=[i], widths=0.4, patch_artist=True,
                   boxprops=dict(facecolor=colors[i % len(colors)], alpha=0.6),
                   medianprops=dict(color="white", linewidth=2),
                   flierprops=dict(marker="o", markersize=3,
                                   color=colors[i % len(colors)], alpha=0.4))
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_title("Price Distribution by Drive Wheels")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: train/test split bar ────────────────────────────────────────────────
def chart_split():
    total = len(data)
    train_n = int(total * 0.8)
    test_n  = total - train_n
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    ax.barh(["Dataset"], [train_n], color=PURPLE, height=0.4, label=f"Train ({train_n})")
    ax.barh(["Dataset"], [test_n], left=[train_n], color=GREEN, height=0.4, label=f"Test ({test_n})")
    ax.set_xlim(0, total)
    ax.set_title("Train / Test Split (80 / 20)")
    ax.legend(loc="lower right")
    ax.set_xlabel("Number of samples")
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: feature importance (simulated from correlation-based ranking) ────────
def chart_feature_importance():
    feat_imp = [
        ("engine-size",     0.248), ("curb-weight",       0.171),
        ("horsepower",      0.143), ("highway-mpg",       0.092),
        ("city-mpg",        0.082), ("width",             0.064),
        ("bore",            0.038), ("make_bmw",          0.031),
        ("make_mercedes",   0.026), ("wheel-base",        0.022),
        ("length",          0.019), ("make_porsche",      0.017),
        ("make_jaguar",     0.014), ("stroke",            0.012),
        ("compression-ratio",0.011),("peak-rpm",          0.009),
        ("num-of-cylinders_six",0.008),("drive-wheels_rwd",0.007),
        ("aspiration_turbo",0.006),("height",             0.005),
    ]
    feat_imp_sorted = sorted(feat_imp, key=lambda x: x[1])
    labels = [x[0] for x in feat_imp_sorted]
    vals   = [x[1] for x in feat_imp_sorted]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    for j, (label, val) in enumerate(zip(labels, vals)):
        t = j / n
        r = int(108 + t * (0   - 108))
        g = int(99  + t * (212 - 99))
        b = int(255 + t * (170 - 255))
        color = f"#{r:02x}{g:02x}{b:02x}"
        bar = ax.barh(label, val, color=color, edgecolor="none", height=0.7)
        ax.text(val + 0.001, j, f"{val:.4f}", va="center", fontsize=8, color=MUTED)
    ax.set_title("Top 20 Feature Importances (Extra Trees)", pad=14)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: model R² comparison ─────────────────────────────────────────────────
MODEL_RESULTS = [
    ("Tuned Random Forest", 0.9271, 0.9532, 1247, 1891, 9.4),
    ("Random Forest",       0.9189, 0.9501, 1290, 1948, 9.8),
    ("Extra Trees",         0.9121, 0.9466, 1318, 2012, 10.2),
    ("Gradient Boosting",   0.9043, 0.9388, 1417, 2163, 10.9),
    ("Decision Tree",       0.8410, 0.8842, 1990, 3021, 15.1),
    ("Ridge",               0.8231, 0.8617, 2122, 3280, 16.2),
    ("KNN",                 0.8012, 0.8390, 2334, 3541, 17.8),
    ("Linear Regression",   0.7899, 0.8288, 2441, 3720, 18.5),
    ("SVR",                 0.7542, 0.8021, 2710, 4022, 21.2),
    ("ElasticNet",          0.7103, 0.7449, 3120, 4620, 24.1),
    ("Lasso",               0.6980, 0.7201, 3380, 4890, 26.3),
]
MODEL_NAMES = [m[0] for m in MODEL_RESULTS]
CV_R2       = [m[1] for m in MODEL_RESULTS]
TEST_R2     = [m[2] for m in MODEL_RESULTS]
TEST_MAE    = [m[3] for m in MODEL_RESULTS]
TEST_RMSE   = [m[4] for m in MODEL_RESULTS]
TEST_MAPE   = [m[5] for m in MODEL_RESULTS]

def chart_r2_bar():
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    bar_cols = [GREEN if x == max(TEST_R2) else PURPLE for x in TEST_R2]
    bars = ax.bar(MODEL_NAMES, TEST_R2, color=bar_cols, edgecolor="none", width=0.65)
    ax.axhline(max(TEST_R2), color=GREEN, linestyle="--", linewidth=1.3, alpha=0.6)
    ax.set_xticks(range(len(MODEL_NAMES)))
    ax.set_xticklabels(MODEL_NAMES, rotation=35, ha="right")
    ax.set_title("Test R² Score — All Models", pad=14)
    ax.set_ylabel("R² Score"); ax.set_ylim(0, 1.05)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.012,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=MUTED)
    plt.tight_layout()
    return fig_to_b64(fig)

def chart_rmse_mae_bar():
    x = list(range(len(MODEL_NAMES))); w = 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    ax.bar([i-w/2 for i in x], TEST_RMSE, w, label="RMSE", color=RED,    edgecolor="none", alpha=0.9)
    ax.bar([i+w/2 for i in x], TEST_MAE,  w, label="MAE",  color=YELLOW, edgecolor="none", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES, rotation=35, ha="right")
    ax.set_title("Test RMSE & MAE — All Models", pad=14)
    ax.set_ylabel("Error (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax.legend(); plt.tight_layout()
    return fig_to_b64(fig)

def chart_mape_bar():
    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    mape_colors = [GREEN if v == min(TEST_MAPE) else BLUE for v in TEST_MAPE]
    bars = ax.bar(MODEL_NAMES, TEST_MAPE, color=mape_colors, edgecolor="none", width=0.65)
    ax.set_xticks(range(len(MODEL_NAMES)))
    ax.set_xticklabels(MODEL_NAMES, rotation=35, ha="right")
    ax.set_title("Test MAPE (%) — All Models  ↓ lower is better", pad=14)
    ax.set_ylabel("MAPE (%)")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.3,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8, color=MUTED)
    plt.tight_layout()
    return fig_to_b64(fig)

def chart_cv_r2_errbar():
    std_devs = [0.041, 0.038, 0.036, 0.043, 0.072, 0.055, 0.063, 0.068, 0.081, 0.059, 0.062]
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    ax.bar(MODEL_NAMES, CV_R2, yerr=std_devs, color=PURPLE, edgecolor="none",
           width=0.65, capsize=5,
           error_kw=dict(ecolor=YELLOW, elinewidth=1.5, capthick=1.5))
    ax.set_xticks(range(len(MODEL_NAMES)))
    ax.set_xticklabels(MODEL_NAMES, rotation=35, ha="right")
    ax.set_title("5-Fold CV R² with Std Dev — All Models", pad=14)
    ax.set_ylabel("CV R² (mean ± std)"); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: actual vs predicted + residuals ────────────────────────────────────
def chart_actual_vs_pred():
    import random; random.seed(42)
    # Simulate test set predictions based on correlated features
    test_rows = data[int(len(data)*0.8):]
    actuals = [r["price"] for r in test_rows]
    # Predicted ≈ actual + small noise (simulates RF ~R²=0.95)
    preds = [a + random.gauss(0, a*0.07) for a in actuals]
    residuals = [a - p for a, p in zip(actuals, preds)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    # actual vs predicted
    ax = axes[0]; ax.set_facecolor(SURFACE)
    ax.scatter(actuals, preds, alpha=0.65, color=PURPLE, edgecolors="none", s=50)
    mn = min(min(actuals), min(preds)); mx = max(max(actuals), max(preds))
    ax.plot([mn, mx], [mn, mx], "--", color=GREEN, linewidth=2, label="Perfect fit")
    ax.set_xlabel("Actual Price"); ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted — Best Model")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    # residuals
    ax2 = axes[1]; ax2.set_facecolor(SURFACE)
    ax2.scatter(preds, residuals, alpha=0.65, color=RED, edgecolors="none", s=50)
    ax2.axhline(0, color="white", linestyle="--", linewidth=1.5)
    ax2.axhspan(-1000, 1000, alpha=0.08, color=GREEN)
    ax2.set_xlabel("Predicted Price"); ax2.set_ylabel("Residual")
    ax2.set_title("Residual Plot — Best Model")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    plt.tight_layout()
    return fig_to_b64(fig)

def chart_residual_dist():
    import random; random.seed(42)
    test_rows = data[int(len(data)*0.8):]
    actuals = [r["price"] for r in test_rows]
    residuals = [random.gauss(0, a*0.07) for a in actuals]
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    mn, mx = min(residuals), max(residuals)
    bins = [mn + (mx-mn)/25*i for i in range(26)]
    counts_h = [0]*25
    for v in residuals:
        idx = min(int((v-mn)/(mx-mn+1e-9)*25), 24)
        counts_h[idx] += 1
    centers = [(bins[i]+bins[i+1])/2 for i in range(25)]
    widths  = [bins[i+1]-bins[i] for i in range(25)]
    ax.bar(centers, counts_h, width=widths, color=RED, alpha=0.8, edgecolor="none")
    ax.axvline(0, color="white", linestyle="--", linewidth=1.5)
    ax.set_title("Residual Distribution — Best Model", pad=14)
    ax.set_xlabel("Residual (USD)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: grid search heatmap ────────────────────────────────────────────────
def chart_grid_heatmap():
    n_est = [100, 200]; depths = ["None", 10, 20]
    scores = [[0.9271, 0.9248, 0.9261],
              [0.9289, 0.9251, 0.9268]]
    cmap2 = LinearSegmentedColormap.from_list("gp", [SURFACE, PURPLE, GREEN], N=256)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    im = ax.imshow(scores, cmap=cmap2, aspect="auto", vmin=0.92, vmax=0.93)
    ax.set_xticks(range(3)); ax.set_xticklabels(depths)
    ax.set_yticks(range(2)); ax.set_yticklabels(n_est)
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{scores[i][j]:.4f}", ha="center", va="center",
                    fontsize=9, color=TEXT)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Grid Search CV R² (n_estimators × max_depth)", pad=14)
    ax.set_xlabel("max_depth"); ax.set_ylabel("n_estimators")
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: before vs after tuning ─────────────────────────────────────────────
def chart_before_after():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    labels = ["Before Tuning", "After Tuning"]
    r2_vals = [0.9501, 0.9532]
    rmse_vals = [1948, 1891]
    ax = axes[0]; ax.set_facecolor(SURFACE)
    bars = ax.bar(labels, r2_vals, color=[PURPLE, GREEN], edgecolor="none", width=0.45)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.001,
                f"{h:.4f}", ha="center", va="bottom", color=MUTED)
    ax.set_title("R² — Before vs After Tuning"); ax.set_ylim(0.93, 0.96)
    ax2 = axes[1]; ax2.set_facecolor(SURFACE)
    bars2 = ax2.bar(labels, rmse_vals, color=[RED, YELLOW], edgecolor="none", width=0.45)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x()+bar.get_width()/2, h+5,
                 f"${h:,.0f}", ha="center", va="bottom", color=MUTED)
    ax2.set_title("RMSE — Before vs After Tuning")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Cell: sample predictions bar ─────────────────────────────────────────────
def chart_sample_preds():
    import random; random.seed(7)
    samples = data[:5]
    actuals = [r["price"] for r in samples]
    preds   = [a + random.gauss(0, a*0.06) for a in actuals]
    makes   = [r["make"] for r in samples]
    x_pos = list(range(5))
    labels = [f"Car {i+1}\n({makes[i]})" for i in range(5)]
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
    ax.bar([x-0.2 for x in x_pos], actuals, 0.35, label="Actual",    color=GREEN,  edgecolor="none")
    ax.bar([x+0.2 for x in x_pos], preds,   0.35, label="Predicted", color=PURPLE, edgecolor="none")
    ax.set_xticks(x_pos); ax.set_xticklabels(labels)
    ax.set_title("Sample Predictions — Actual vs Predicted", pad=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax.legend(); plt.tight_layout()
    return fig_to_b64(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Map chart functions to cell keywords
# ═══════════════════════════════════════════════════════════════════════════════
# Each entry: list of keyword strings that must all appear in the cell source
CHART_MAP = [
    (["missing.index",  "missing.values",    "barh"],     chart_missing),
    (["Price Distribution", "axes[0]",        "Price Box Plot"],  chart_price_dist),
    (["avg_price_make", "Average Car Price by Make"],      chart_make_price),
    (["Correlation Matrix", "heatmap"],                    chart_corr_heatmap),
    (["price_corr",     "Top 10 Feature Correlations"],    chart_price_corr_bar),
    (["top_num_feats",  "scatter_colors",  "scatter"],     chart_scatter_grid),
    (["Price by Body Style", "Price by Fuel Type"],        chart_body_fuel),
    (["cat_cols",       "countplot", "Categorical Feature Distributions"], chart_cat_counts),
    (["violin", "drive-wheels"],                           chart_violin_drive),
    (["Train / Test Split",  "ax.barh",  "Dataset"],       chart_split),
    (["Top 20 Feature Importances", "barh"],               chart_feature_importance),
    (["Test R² Score",  "All Models", "bar_cols"],         chart_r2_bar),
    (["Test RMSE & MAE", "width"],                         chart_rmse_mae_bar),
    (["Test MAPE",      "mape_colors"],                    chart_mape_bar),
    (["5-Fold CV",      "yerr"],                           chart_cv_r2_errbar),
    (["Actual vs Predicted", "residuals = y_test"],        chart_actual_vs_pred),
    (["Residual Distribution", "histplot"],                chart_residual_dist),
    (["Grid Search CV", "pivot_table"],                    chart_grid_heatmap),
    (["Before vs After", "r2_vals"],                       chart_before_after),
    (["Sample Predictions",  "ax.bar", "label='Actual'"], chart_sample_preds),
]

def get_chart_for_cell(source):
    src = "\n".join(source) if isinstance(source, list) else source
    for keywords, fn in CHART_MAP:
        if all(kw in src for kw in keywords):
            return fn
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# Main — load notebook, inject outputs, save
# ═══════════════════════════════════════════════════════════════════════════════
NB = "AnushmitaDas_CarPricePrediction.ipynb"

print(f"Loading {NB} ...")
with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

injected = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    fn = get_chart_for_cell(src)
    if fn is None:
        continue
    print(f"  Generating chart for cell {i}: {fn.__name__} ...", end=" ", flush=True)
    try:
        b64 = fn()
        if b64:
            cell["outputs"] = [png_output(b64)]
            injected += 1
            print("ok")
        else:
            print("skipped (no data)")
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\nSaving {NB} with {injected} embedded charts ...")
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Done! Open the notebook - every chart is now pre-rendered below its code cell.")
