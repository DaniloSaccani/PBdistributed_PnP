"""Title-free paper figures from raw samples, without smoothing or decimation."""
from pathlib import Path
import numpy as np

VIEWS = {
    "plugin_DGU6": dict(nodes=(1, 5, 6), end=4.25, zoom=(3.995, 4.03), pad=.48,
                        lower=.28, file="paper_plugin_dgu_1_5_6"),
    "post_event_load": dict(nodes=(1, 5, 6), end=8.40, zoom=(7.995, 8.06), pad=.42,
                           lower=.22, file="paper_post_event_load_dgu_1_5_6"),
    "unplug_DGU3": dict(nodes=(1, 4), end=12.25, zoom=(11.995, 12.06), pad=.42,
                       lower=.22, file="paper_unplug_dgu3_dgu_1_4"),
}


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _padded(values, minimum, fraction):
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("No active samples in this plot window")
    lo, hi = float(finite.min()), float(finite.max())
    span, center = max(hi-lo, minimum), (lo+hi)/2
    return center-(.5+fraction)*span, center+(.5+fraction)*span


def render_case(scenario, baseline, mad, registry, output, *, controls=False, legend=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    view = VIEWS[scenario.name]
    traces = [{key: _array(trace[key]) for key in ("X", "active", "u")}
              for trace in (baseline, mad)]
    n = scenario.steps+1
    if any(trace["X"].shape != (n, registry.state_dim) or
           trace["active"].shape != (n, registry.input_dim) or
           trace["u"].shape != (n-1, registry.input_dim) for trace in traces):
        raise ValueError("Baseline and PB must use the same complete scenario")
    t = scenario.time_origin_s + np.arange(n)*scenario.plant.h
    event_time = scenario.time_origin_s + scenario.event_sample*scenario.plant.h
    full = (t >= max(t[0], event_time-.05)) & (t <= view["end"])
    zoom = (t >= view["zoom"][0]) & (t <= view["zoom"][1])
    positions = {node: j for j, node in enumerate(registry.node_ids)}
    references = {node.id: node.Vref for node in registry.nodes}
    voltages = {}
    for node in view["nodes"]:
        j = positions[node]
        voltages[node] = []
        for trace in traces:
            v = trace["X"][:, 3*j].copy()
            v[~trace["active"][:, j].astype(bool)] = np.nan
            voltages[node].append(v)
    ylo, yhi = _padded(np.concatenate([v[full] for pair in voltages.values() for v in pair]),
                        .22, view["pad"])
    style = {"path.simplify": False, "font.family": "sans-serif", "font.size": 11,
             "axes.grid": True, "grid.color": ".8", "grid.linewidth": .8, "pdf.fonttype": 42}
    output = Path(output)
    paths = []
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, len(view["nodes"]),
                                figsize=(4.8*len(view["nodes"]), 3.55), squeeze=False)
        for ax, node in zip(axes[0], view["nodes"]):
            base_v, mad_v = voltages[node]
            ax.plot(t[full], mad_v[full], color="#4C72B0", lw=2.2, label="PB + MAD")
            ax.plot(t[full], base_v[full], color="#C44E52", ls="--", lw=2.2, label="Baseline")
            ax.axhline(references[node], color="black", ls=":", lw=.9, label="Reference")
            ax.axvline(event_time, color="black", ls=":", lw=1.)
            ax.set(xlim=(t[full][0], t[full][-1]), ylim=(ylo-view["lower"], yhi),
                   ylabel=r"$V_{%d}$ [V]" % node, xlabel="time [s]")
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            inset = inset_axes(ax, width="48%", height="48%", loc="lower right", borderpad=1.)
            inset.plot(t[zoom], mad_v[zoom], color="#4C72B0", lw=1.5)
            inset.plot(t[zoom], base_v[zoom], color="#C44E52", ls="--", lw=1.5)
            inset.axhline(references[node], color="black", ls=":", lw=.7)
            inset.axvline(event_time, color="black", ls=":", lw=.8)
            inset.set_xlim(t[zoom][0], t[zoom][-1])
            inset.set_ylim(*_padded(np.concatenate([base_v[zoom], mad_v[zoom], [references[node]]]), .055, .16))
            inset.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
            mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec=".35", lw=.8)
        if legend:
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.subplots_adjust(left=.07, right=.975, bottom=.18, top=.87, wspace=.30)
        for ext in ("pdf", "png"):
            path = output / (view["file"] + "." + ext)
            save_options = {} if legend else {"bbox_inches": "tight", "pad_inches": .05}
            fig.savefig(path, dpi=220, **save_options)
            paths.append(path)
        plt.close(fig)
        if controls:
            fig, axes = plt.subplots(1, len(view["nodes"]),
                                    figsize=(4.8*len(view["nodes"]), 3.55), squeeze=False)
            control_t = t[:-1]
            show = full[:-1]
            fast = (control_t >= event_time-.001) & (control_t <= event_time+.010)
            for ax, node in zip(axes[0], view["nodes"]):
                j = positions[node]
                u = traces[1]["u"][:, j]
                ax.plot(control_t[show], u[show], color="#4C72B0", lw=1.8)
                ax.axvline(event_time, color="black", ls=":", lw=1.)
                ax.set(xlim=(control_t[show][0], control_t[show][-1]),
                       xlabel="time [s]", ylabel=r"$\Delta u_{%d}$ [V]" % node)
                ax.ticklabel_format(axis="x", style="plain", useOffset=False)
                inset = inset_axes(ax, width="48%", height="48%", loc="upper right", borderpad=1.)
                inset.plot((control_t[fast]-event_time)*1000, u[fast], color="#4C72B0", lw=1.2)
                inset.axvline(0, color="black", ls=":", lw=.8)
                inset.tick_params(labelsize=8)
                inset.set_xlabel("after event [ms]", fontsize=8)
            fig.subplots_adjust(left=.09, right=.975, bottom=.18, top=.94, wspace=.35)
            for ext in ("pdf", "png"):
                path = output / (view["file"] + "_control." + ext)
                fig.savefig(path, dpi=220)
                paths.append(path)
            plt.close(fig)
    return paths
