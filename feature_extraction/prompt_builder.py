from __future__ import annotations

from core.schema import TSFactCard


def build_fact_card_text(card: TSFactCard) -> str:
    lines = [
        f"TS-Fact Card：窗口 {card.window_id}，长度 {card.length}，变量数 {len(card.variables)}。",
    ]
    if card.trends:
        top = sorted(card.trends, key=lambda item: abs(item.tau), reverse=True)[:4]
        desc = [f"{t.variable} 呈{t.strength}{t.direction}趋势(Kendall tau={t.tau:.3f})" for t in top]
        lines.append("趋势：" + "；".join(desc) + "。")
    if card.periodicities:
        sig = [p for p in card.periodicities if p.significant][:4]
        if sig:
            desc = []
            for p in sig:
                period = f"主周期约 {p.dominant_period:.1f}" if p.dominant_period else f"ACF峰值滞后 {p.acf_peak_lag}"
                desc.append(f"{p.variable} {period}，FFT能量占比 {p.fft_power_ratio:.2f}")
            lines.append("周期：" + "；".join(desc) + "。")
        else:
            lines.append("周期：未检测到显著稳定周期。")
    if card.anomaly_profile:
        p = card.anomaly_profile
        lines.append(
            f"异常剖面：高分点密度 {p.density:.2%}，簇数量 {p.cluster_count}，峰值位置 {p.peak_position}，{p.temporal_distribution}。"
        )
    if card.correlations:
        desc = []
        for c in card.correlations[:4]:
            change = "，相关结构发生明显变化" if c.changed else ""
            desc.append(f"{c.variable_a}-{c.variable_b} Pearson={c.pearson:.3f}{change}")
        lines.append("变量相关性：" + "；".join(desc) + "。")
    if card.residuals:
        top_res = sorted(card.residuals, key=lambda item: item.max_abs_error, reverse=True)[:3]
        desc = [f"{r.variable} 平均偏差 {r.mean_bias:.3f}，最大绝对误差 {r.max_abs_error:.3f}" for r in top_res]
        lines.append("预测误差：" + "；".join(desc) + "。")
    if card.attributions:
        desc = [f"{a.variable} 贡献约 {a.contribution:.1%}({a.reason})" for a in card.attributions[:4]]
        lines.append("变量贡献：" + "；".join(desc) + "。")
    if card.changepoints:
        lines.append("变点：检测到候选变点位置 " + "、".join(map(str, card.changepoints)) + "。")
    if card.stationarity:
        unstable = [name for name, info in card.stationarity.items() if not info.get("stationary_hint", True)]
        if unstable:
            lines.append("平稳性：以下变量存在非平稳迹象：" + "、".join(unstable[:6]) + "。")
    return "\n".join(lines)

