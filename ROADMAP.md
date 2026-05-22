# Roadmap

Tracked, deliberately-not-yet-scheduled ideas for the dashboard. Each item links
to a GitHub issue where the thinking and discussion live.

## Tabled / future

### Global expansion — rotating-globe landing page + per-country dashboards
**Status:** captured, tabled (not scheduled) · **Issue:** [#35](https://github.com/SecondOrderEdge/Macro-Dashboard/issues/35)

A slowly rotating 3D globe as the landing page, shaded by each country's current
recession-risk / growth signal (the globe *is* the hero heatmap, not just
navigation). Click a country → that country's macro dashboard, replicating the US
dashboard per country over time.

Key points (full detail in the issue):
- The hard part is **data + models, not the globe UI** — the whole stack is
  US-wired (FRED IDs, NBER target, Treasury curve, Atlanta Fed policy path,
  Shiller CAPE, US-calibrated probit).
- **Tiered honesty:** full treatment only for data-rich economies; a
  recession-risk *lite* tier; and a snapshot tier elsewhere — never fake rigor.
- **Sequence:** generalize the data layer → prove with one second country
  end-to-end (Canada/UK) → *then* build the globe → expand. Resist building the
  globe first.
- **Open:** showcase piece vs. analytical tool? country #2? scaling via a
  precomputed nightly pipeline rather than live FRED-on-load.
