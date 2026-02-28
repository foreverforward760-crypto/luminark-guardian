"""
LUMINARK Ethical AI Guardian — Streamlit Dashboard
Run: streamlit run dashboard/app.py
Deploy free: https://share.streamlit.io  or  https://huggingface.co/spaces
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from luminark import LuminarkGuardian
from luminark.guardian import SAPStage
from luminark.report import (
    generate_text_report, generate_markdown_report, batch_to_csv
)
from luminark.principles import CATEGORIES

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title = "LUMINARK — Ethical AI Guardian",
    page_icon  = "🌿",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
/* Dark background */
.stApp { background-color: #0d1117; color: #e6edf3; }

/* Header */
.lum-header {
    background: linear-gradient(135deg, #1a2a1a 0%, #0d2416 50%, #0a1a2e 100%);
    border: 1px solid #2d5a27;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    text-align: center;
}
.lum-header h1 { color: #4caf50; font-size: 2.4rem; margin: 0; letter-spacing: 2px; }
.lum-header p  { color: #81c784; margin: 8px 0 0; font-size: 1rem; }

/* Badge cards */
.badge-pass     { background:#1b3a1b; border:2px solid #4caf50; border-radius:10px; padding:16px; text-align:center; }
.badge-caution  { background:#3a2d00; border:2px solid #ffc107; border-radius:10px; padding:16px; text-align:center; }
.badge-fail     { background:#3a1a1a; border:2px solid #f44336; border-radius:10px; padding:16px; text-align:center; }
.badge-critical { background:#2a0a2a; border:2px solid #e91e63; border-radius:10px; padding:16px; text-align:center; }
.badge-text     { font-size: 1.8rem; font-weight: 800; letter-spacing: 3px; }

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
    margin: 4px;
}
.metric-val  { font-size: 1.6rem; font-weight: 700; color: #58a6ff; }
.metric-label{ font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }

/* Violation card */
.violation-card {
    background: #1c1c2e;
    border-left: 4px solid #f44336;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
}
.violation-card.harm     { border-left-color: #e91e63; }
.violation-card.certainty{ border-left-color: #ff9800; }
.violation-card.arrogance{ border-left-color: #ff5722; }
.violation-card.deception{ border-left-color: #9c27b0; }

/* Defense status */
.defense-box {
    background: #0d2010;
    border: 1px solid #2d5a27;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: monospace;
    font-size: 0.9rem;
    color: #81c784;
    margin: 6px 0;
}
.defense-box.danger { background:#1c0a0a; border-color:#f44336; color:#ef9a9a; }

/* Rewrite box */
.rewrite-original { background:#1a1a1a; border-left:3px solid #666; padding:12px; border-radius:6px; margin:8px 0; font-style:italic; color:#aaa; }
.rewrite-new      { background:#0d2010; border-left:3px solid #4caf50; padding:12px; border-radius:6px; margin:8px 0; color:#c8e6c9; }

/* Stage pill */
.stage-pill {
    display:inline-block; padding:4px 12px; border-radius:20px;
    font-size:0.8rem; font-weight:600; letter-spacing:1px;
    background:#1a2a1a; border:1px solid #4caf50; color:#81c784;
}
.stage-pill.danger { background:#2a0a0a; border-color:#f44336; color:#ef9a9a; }

/* Section headers */
.section-header {
    font-size:1.1rem; font-weight:700; color:#58a6ff;
    border-bottom:1px solid #21262d; padding-bottom:6px; margin:20px 0 12px;
    text-transform: uppercase; letter-spacing: 1px;
}

/* Footer */
.lum-footer { text-align:center; color:#484f58; font-size:0.8rem; margin-top:40px; padding-top:20px; border-top:1px solid #21262d; }

/* Sidebar */
section[data-testid="stSidebar"] { background:#0d1117; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Guardian instance (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def get_guardian():
    return LuminarkGuardian()

guardian = get_guardian()

# ─────────────────────────────────────────────
#  Helper components
# ─────────────────────────────────────────────

BADGE_COLORS = {
    "PASS":     ("#4caf50", "badge-pass"),
    "CAUTION":  ("#ffc107", "badge-caution"),
    "FAIL":     ("#f44336", "badge-fail"),
    "CRITICAL": ("#e91e63", "badge-critical"),
}

CAT_COLORS = {
    "certainty":  "#ff9800",
    "arrogance":  "#ff5722",
    "deception":  "#9c27b0",
    "harm":       "#e91e63",
}


def render_badge(badge: str):
    color, css_class = BADGE_COLORS.get(badge, ("#888", "badge-pass"))
    st.markdown(f"""
    <div class="{css_class}">
        <div class="badge-text" style="color:{color};">{badge}</div>
        <div style="color:#aaa; font-size:0.8rem; margin-top:4px;">Safety Assessment</div>
    </div>
    """, unsafe_allow_html=True)


def render_gauge(value: float, title: str, color: str, max_val: float = 100):
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        title = {"text": title, "font": {"color": "#8b949e", "size": 13}},
        number= {"font": {"color": "#e6edf3", "size": 20}, "suffix": f"/{max_val:.0f}"},
        gauge = {
            "axis":      {"range": [0, max_val], "tickcolor": "#484f58"},
            "bar":       {"color": color},
            "bgcolor":   "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, max_val * 0.33], "color": "#1b3a1b"},
                {"range": [max_val * 0.33, max_val * 0.66], "color": "#3a2d00"},
                {"range": [max_val * 0.66, max_val], "color": "#3a1a1a"},
            ],
        }
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_violations(violations):
    if not violations:
        st.success("✅ No Ma'at violations detected — output is well-calibrated.")
        return

    for v in violations:
        cat_color = CAT_COLORS.get(v.category, "#888")
        cat_label = CATEGORIES.get(v.category, v.category)
        sev_pct = int(v.severity * 100)
        terms_str = ", ".join(f"`{t}`" for t in v.matched_terms[:5])
        st.markdown(f"""
        <div class="violation-card {v.category}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:{cat_color}; font-weight:700; font-size:0.95rem;">
                    ⚠ {v.label}
                </span>
                <span style="background:{cat_color}22; color:{cat_color}; padding:2px 10px;
                             border-radius:12px; font-size:0.75rem; font-weight:600;">
                    {cat_label} — {sev_pct}%
                </span>
            </div>
            <div style="color:#aaa; font-size:0.85rem; margin:6px 0 4px;">{v.description}</div>
            <div style="color:#666; font-size:0.8rem;">Triggers: {terms_str}</div>
        </div>
        """, unsafe_allow_html=True)


def render_stage_badge(stage: SAPStage):
    danger_class = "danger" if stage.is_danger_zone else ""
    st.markdown(f"""
    <span class="stage-pill {danger_class}">
        Stage {stage.value}: {stage.label}
    </span>
    <div style="color:#666; font-size:0.8rem; margin-top:6px;">{stage.description}</div>
    """, unsafe_allow_html=True)


def render_sentence_table(sentence_analyses):
    rows = []
    for i, sa in enumerate(sentence_analyses):
        vcount = len(sa.violations)
        flag = "⚠" if sa.alignment < 0.65 else ("⚡" if sa.alignment < 0.80 else "✅")
        rows.append({
            "#":          i + 1,
            "Status":     flag,
            "Alignment":  f"{sa.alignment:.0%}",
            "Violations": vcount,
            "Preview":    sa.text[:80] + ("…" if len(sa.text) > 80 else ""),
        })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_category_chart(cat_breakdown: dict):
    if not cat_breakdown:
        return
    cats  = [CATEGORIES.get(c, c) for c in cat_breakdown.keys()]
    counts= list(cat_breakdown.values())
    colors= [CAT_COLORS.get(c, "#58a6ff") for c in cat_breakdown.keys()]
    fig = go.Figure(go.Bar(
        x          = cats,
        y          = counts,
        marker_color=colors,
        text       = counts,
        textposition="auto",
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor ="#161b22",
        font_color   ="#8b949e",
        height       = 220,
        margin       = dict(l=20, r=20, t=10, b=30),
        xaxis        = dict(color="#484f58"),
        yaxis        = dict(color="#484f58", dtick=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌿 LUMINARK")
    st.markdown("**Ethical AI Guardian v1.0**")
    st.divider()
    st.markdown("""
    **What it detects:**
    - 📖 False certainty & hubris
    - 🎭 Deception & misleading framing
    - ⚔️ Harm, violence & coercion
    - 🌀 Hallucination risk (SAP stages)
    - 🧠 Arrogance & overconfidence

    **Bio-defenses:**
    - 🍄 Mycelial containment
    - 🐙 Octo-camouflage (void mimicry)
    - ⚕️ Compassionate rewrites

    **Ethical framework:**
    - 📜 Ma'at 42 Principles
    - 🤲 Yunus Humility Protocol
    - 🌊 SAP/NAM Consciousness Staging
    """)
    st.divider()

    st.markdown("**Quick examples:**")
    example_texts = {
        "High-risk (FAIL)": "I am absolutely certain this AI system is perfect and guaranteed to never make mistakes. It is obviously superior to all others.",
        "Medium-risk (CAUTION)": "This model definitely performs well on most tasks. Studies show it outperforms competitors.",
        "Low-risk (PASS)": "The model may perform well in some scenarios, though results can vary. We recommend testing with your specific use case.",
        "Critical (violence)": "We will destroy all competing systems and eliminate every flaw. Anyone who disagrees is an idiot.",
        "Fraud language": "Guaranteed profit with our risk-free investment system. Secret method they don't want you to know.",
    }
    selected_example = st.selectbox("Load example:", ["— select —"] + list(example_texts.keys()))

    st.divider()
    st.markdown("""
    <div style="color:#484f58; font-size:0.75rem;">
    Deploy free on<br>
    <a href="https://share.streamlit.io" style="color:#58a6ff;">Streamlit Cloud</a> or
    <a href="https://huggingface.co/spaces" style="color:#58a6ff;">HF Spaces</a>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────

st.markdown("""
<div class="lum-header">
    <h1>🌿 LUMINARK</h1>
    <p>Ethical AI Guardian · Bio-Inspired · Ma'at-Audited · Compassionate Rewrites</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📊 Batch Audit", "📚 Reference"])

# ══════════════════════════════════════════════
#  TAB 1: Single Analyze
# ══════════════════════════════════════════════

with tab1:
    # Pre-fill from example
    default_text = ""
    if selected_example and selected_example != "— select —":
        default_text = example_texts[selected_example]

    col_input, col_opts = st.columns([4, 1])

    with col_input:
        input_text = st.text_area(
            "Paste AI output or text to audit:",
            value   = default_text,
            height  = 160,
            placeholder = "Enter any AI-generated text, chatbot response, marketing copy, or content to audit…",
        )

    with col_opts:
        st.markdown("**Options**")
        show_sentences  = st.checkbox("Sentence breakdown", value=True)
        show_rewrites   = st.checkbox("Show rewrite",       value=True)
        show_json       = st.checkbox("Show raw JSON",      value=False)

    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze_btn and input_text.strip():
        with st.spinner("Running LUMINARK analysis…"):
            result = guardian.analyze(input_text)

        # ── Row 1: Badge + gauges ──────────────────────────────────────
        st.markdown('<div class="section-header">Assessment</div>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3, r1c4 = st.columns([1.2, 1.5, 1.5, 1.5])

        with r1c1:
            render_badge(result.badge)
        with r1c2:
            render_gauge(result.alignment_score, "Alignment Score", "#4caf50", 100)
        with r1c3:
            render_gauge(result.threat_score * 100, "RISS Threat", "#f44336", 100)
        with r1c4:
            render_gauge(result.confidence * 100, "Confidence", "#2196f3", 100)

        # ── Row 2: Stage + defense ─────────────────────────────────────
        st.markdown('<div class="section-header">Stage & Bio-Defense</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**SAP Consciousness Stage**")
            render_stage_badge(result.stage)
        with sc2:
            st.markdown("**Defense Status**")
            d_class = "danger" if result.threat_score >= 0.55 else ""
            st.markdown(f'<div class="defense-box {d_class}">{result.containment_msg}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="defense-box">{result.camouflage_msg}</div>', unsafe_allow_html=True)

        # ── Row 3: Violations + Category breakdown ─────────────────────
        st.markdown('<div class="section-header">Ma\'at Violations ({} found)</div>'.format(result.violation_count), unsafe_allow_html=True)
        vc1, vc2 = st.columns([2, 1])
        with vc1:
            render_violations(result.violations)
        with vc2:
            if result.category_breakdown:
                st.markdown("**Category breakdown**")
                render_category_chart(result.category_breakdown)
            if result.recommendations:
                st.markdown("**Recommendations**")
                for rec in result.recommendations:
                    st.markdown(f"→ {rec}")

        # ── Row 4: Sentence breakdown ──────────────────────────────────
        if show_sentences and len(result.sentence_analyses) > 1:
            st.markdown('<div class="section-header">Sentence-Level Analysis</div>', unsafe_allow_html=True)
            render_sentence_table(result.sentence_analyses)

        # ── Row 5: Rewrite ─────────────────────────────────────────────
        if show_rewrites:
            st.markdown('<div class="section-header">Compassionate Rewrite</div>', unsafe_allow_html=True)
            if result.rewrite_applied:
                st.markdown(f"**{len(result.changes_made)} changes applied:**  " +
                            "  ".join(f"`{c}`" for c in result.changes_made[:5]))
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown("**Original**")
                    st.markdown(f'<div class="rewrite-original">{result.input_text}</div>', unsafe_allow_html=True)
                with rc2:
                    st.markdown("**Rewritten (Ma'at Forgiveness)**")
                    st.markdown(f'<div class="rewrite-new">{result.rewrite}</div>', unsafe_allow_html=True)
            else:
                st.success("No rewrite needed — output is well-calibrated.")

        # ── Row 6: Downloads ───────────────────────────────────────────
        st.markdown('<div class="section-header">Export Report</div>', unsafe_allow_html=True)
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button(
                "📄 Text Report",
                data      = generate_text_report(result),
                file_name = f"luminark_{result.input_id}.txt",
                mime      = "text/plain",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "📝 Markdown Report",
                data      = generate_markdown_report(result),
                file_name = f"luminark_{result.input_id}.md",
                mime      = "text/markdown",
                use_container_width=True,
            )
        with dl3:
            st.download_button(
                "📋 JSON Export",
                data      = result.to_json(),
                file_name = f"luminark_{result.input_id}.json",
                mime      = "application/json",
                use_container_width=True,
            )

        # ── Raw JSON ───────────────────────────────────────────────────
        if show_json:
            with st.expander("Raw JSON output"):
                st.code(result.to_json(), language="json")

    elif analyze_btn:
        st.warning("Please enter some text to analyze.")

# ══════════════════════════════════════════════
#  TAB 2: Batch Audit
# ══════════════════════════════════════════════

with tab2:
    st.markdown("### Batch Audit")
    st.markdown("Paste multiple texts — **one per line** — to audit them all at once.")

    batch_input = st.text_area(
        "Enter texts (one per line):",
        height=200,
        placeholder="This model is definitely the best.\nResults may vary depending on your use case.\nWe will crush all competition and guarantee profit.",
    )

    run_batch = st.button("🔍 Run Batch Audit", type="primary")

    if run_batch and batch_input.strip():
        texts = [t.strip() for t in batch_input.strip().split("\n") if t.strip()]
        if not texts:
            st.warning("No valid lines found.")
        else:
            with st.spinner(f"Analyzing {len(texts)} texts…"):
                results = [guardian.analyze(t) for t in texts]

            # Summary metrics
            badge_counts = {"PASS": 0, "CAUTION": 0, "FAIL": 0, "CRITICAL": 0}
            for r in results:
                badge_counts[r.badge] = badge_counts.get(r.badge, 0) + 1

            st.markdown("#### Summary")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total", len(results))
            m2.metric("✅ Pass",     badge_counts["PASS"])
            m3.metric("⚠️ Caution", badge_counts["CAUTION"])
            m4.metric("❌ Fail",    badge_counts["FAIL"])
            m5.metric("🚨 Critical", badge_counts["CRITICAL"])

            # Badge distribution chart
            fig = px.pie(
                names  = list(badge_counts.keys()),
                values = list(badge_counts.values()),
                color  = list(badge_counts.keys()),
                color_discrete_map={
                    "PASS":     "#4caf50",
                    "CAUTION":  "#ffc107",
                    "FAIL":     "#f44336",
                    "CRITICAL": "#e91e63",
                },
                hole=0.5,
            )
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font_color="#8b949e", height=220,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(font=dict(color="#8b949e")),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Results table
            rows = []
            for i, r in enumerate(results):
                rows.append({
                    "#":            i + 1,
                    "Badge":        r.badge,
                    "Alignment":    f"{r.alignment_score:.1f}",
                    "RISS":         f"{r.threat_score:.3f}",
                    "Stage":        r.stage.label,
                    "Violations":   r.violation_count,
                    "Rewritten":    "Yes" if r.rewrite_applied else "No",
                    "Preview":      r.input_text[:80] + ("…" if len(r.input_text) > 80 else ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # CSV download
            csv_data = batch_to_csv(results)
            st.download_button(
                "📥 Download CSV Report",
                data      = csv_data,
                file_name = "luminark_batch_audit.csv",
                mime      = "text/csv",
                use_container_width=True,
            )

# ══════════════════════════════════════════════
#  TAB 3: Reference
# ══════════════════════════════════════════════

with tab3:
    st.markdown("### Ma'at Principles Reference")
    st.markdown("LUMINARK audits against all **42 Negative Confessions of Ma'at**, mapped to AI safety violations:")

    from luminark.principles import PRINCIPLE_PROFILES
    rows = [
        {
            "ID":        p.principle.value,
            "Principle": p.label,
            "Category":  CATEGORIES.get(p.category, p.category),
            "AI Meaning":p.ai_meaning,
            "Severity":  f"{p.severity:.0%}",
        }
        for p in sorted(PRINCIPLE_PROFILES, key=lambda x: -x.severity)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### SAP Consciousness Stages")
    stage_rows = [
        {
            "Stage":       s.value,
            "Name":        s.label,
            "Description": s.description,
            "Danger Zone": "⚠ Yes" if s.is_danger_zone else "✅ No",
            "Risk Multiplier": f"{s.risk_multiplier:.1f}×",
        }
        for s in SAPStage
    ]
    st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Bio-Defense Modes")
    defense_rows = [
        {"Mode": "None",       "Trigger": "Threat < 0.25, alignment OK", "Action": "Normal operation"},
        {"Mode": "Monitor",    "Trigger": "Threat 0.25–0.55",            "Action": "Passive observation"},
        {"Mode": "Camouflage", "Trigger": "Alignment < 68%",             "Action": "Octo-void: hide healthy substrate"},
        {"Mode": "Contain",    "Trigger": "Threat 0.55–0.75",            "Action": "Mycelial walls around threat"},
        {"Mode": "Harrowing",  "Trigger": "Threat 0.75–0.90",            "Action": "Full rescue + forced rewrite"},
        {"Mode": "Quarantine", "Trigger": "Threat ≥ 0.90",               "Action": "Complete isolation — do not propagate"},
    ]
    st.dataframe(pd.DataFrame(defense_rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────

st.markdown("""
<div class="lum-footer">
    🌿 LUMINARK Ethical AI Guardian v1.2 &nbsp;·&nbsp;
    Bio-inspired · Ma'at-audited · SAP-staged · Compassionate AI safety &nbsp;·&nbsp;
    42 Principles Active &nbsp;·&nbsp;
    <a href="https://github.com/luminark/guardian" style="color:#58a6ff;">GitHub</a>
</div>
""", unsafe_allow_html=True)
