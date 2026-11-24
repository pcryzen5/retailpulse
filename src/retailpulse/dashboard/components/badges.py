"""Status badges and risk indicators.

Implements DESIGN §5, §7, and §8:
- Icon + text + color:
    ok ✔, reorder now ⚠, overstock ◔, no data -, high/medium/low risk ●
- Accessibility: Status and risk are never conveyed by color alone.
- Security: Uses native Streamlit colored markdown (:color[text]), zero unsafe_allow_html.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class BadgeDefinition:
    label: str
    icon: str
    color: str  # Streamlit markdown color: green, red, orange, blue, gray

    @property
    def display_text(self) -> str:
        return f"{self.icon} {self.label}".strip()

    @property
    def markdown(self) -> str:
        return f":{self.color}[{self.display_text}]"


BADGE_REGISTRY: dict[str, BadgeDefinition] = {
    # Stock and inventory
    "ok": BadgeDefinition("OK", "✔", "green"),
    "in_stock": BadgeDefinition("OK", "✔", "green"),
    "reorder_now": BadgeDefinition("Reorder now", "⚠", "red"),
    "reorder": BadgeDefinition("Reorder now", "⚠", "red"),
    "overstock": BadgeDefinition("Overstock", "◔", "orange"),
    "no_data": BadgeDefinition("No data", "-", "gray"),
    "none": BadgeDefinition("No data", "-", "gray"),
    # Churn and risk bands
    "high": BadgeDefinition("High risk", "●", "red"),
    "high_risk": BadgeDefinition("High risk", "●", "red"),
    "medium": BadgeDefinition("Medium risk", "●", "orange"),
    "medium_risk": BadgeDefinition("Medium risk", "●", "orange"),
    "low": BadgeDefinition("Low risk", "●", "green"),
    "low_risk": BadgeDefinition("Low risk", "●", "green"),
    # Models and flags
    "synthetic": BadgeDefinition("Synthetic", "🧪", "blue"),
    "champion": BadgeDefinition("Champion", "⭐", "blue"),
    "challenger": BadgeDefinition("Challenger", "⚡", "orange"),
    "active": BadgeDefinition("Active", "●", "green"),
    "inactive": BadgeDefinition("Inactive", "●", "gray"),
    # Pipelines and workflows
    "queued": BadgeDefinition("Queued", "⏳", "blue"),
    "running": BadgeDefinition("Running", "⚡", "orange"),
    "succeeded": BadgeDefinition("Succeeded", "✔", "green"),
    "failed": BadgeDefinition("Failed", "❌", "red"),
}


def get_badge_definition(status: str | None) -> BadgeDefinition:
    """Resolve normalized badge definition for a given status string."""
    if not status:
        return BADGE_REGISTRY["no_data"]

    norm_key = str(status).strip().lower().replace(" ", "_").replace("-", "_")
    return BADGE_REGISTRY.get(norm_key, BadgeDefinition(str(status), "●", "gray"))


def format_status_badge(status: str | None) -> str:
    """Format status as a native colored markdown string."""
    badge = get_badge_definition(status)
    return badge.markdown


def render_badge(status: str | None) -> None:
    """Render status badge into active Streamlit view."""
    st.markdown(format_status_badge(status))
