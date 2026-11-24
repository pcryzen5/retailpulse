"""Server-side paginated data table component.

Implements DESIGN §5 and §8:
- st.dataframe with column_config and sticky headers.
- Server-side pagination controls (Previous / Next, page counter, record bounds).
- Accessible column alignment and formatting.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st


def render_paged_table(
    df: pd.DataFrame,
    *,
    total_count: int,
    page: int,
    page_size: int = 50,
    column_config: dict[str, Any] | None = None,
    key: str = "paged_table",
    hide_index: bool = True,
) -> int:
    """Render a data table with server-side pagination controls.

    Returns:
        The current active page number.
    """
    st.dataframe(
        df,
        column_config=column_config,
        hide_index=hide_index,
        use_container_width=True,
    )

    total_pages = max(1, math.ceil(total_count / max(1, page_size)))
    current_page = max(1, min(page, total_pages))

    start_idx = (current_page - 1) * page_size + 1 if total_count > 0 else 0
    end_idx = min(current_page * page_size, total_count)

    col_info, col_page, col_nav = st.columns([4, 4, 4])

    with col_info:
        st.caption(f"Showing **{start_idx}-{end_idx}** of **{total_count}** items")

    with col_page:
        st.caption(f"Page **{current_page}** of **{total_pages}**")

    with col_nav:
        btn_prev, btn_next = st.columns(2)
        with btn_prev:
            prev_clicked = st.button(
                "◀ Previous",
                key=f"{key}_prev",
                disabled=(current_page <= 1),
                use_container_width=True,
            )
        with btn_next:
            next_clicked = st.button(
                "Next ▶",
                key=f"{key}_next",
                disabled=(current_page >= total_pages),
                use_container_width=True,
            )

        if prev_clicked:
            st.session_state[f"{key}_page"] = current_page - 1
            st.rerun()

        if next_clicked:
            st.session_state[f"{key}_page"] = current_page + 1
            st.rerun()

    return current_page
