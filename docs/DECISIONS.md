# RetailPulse — Architecture and Technical Decisions

This document records architectural and technical decisions made during the design and implementation of RetailPulse.
Format: Date, Decision, Context/Reason, Consequences.

---

## Decisions Log

### 2026-10-16: DEC-001 - Project Workspace & Repository Initialization
- **Date:** 2026-10-16
- **Decision:** Initialized the git repository at workspace root `c:\Users\shubhAM\Desktop\Masters_project`, promoted `.cursorrules`, `CLAUDE.md`, and `docs/` to root, and placed `online_retail_II.xlsx` into `.gitignore` (and planned for `data/raw/`).
- **Context/Reason:** Establishes the single root workspace conforming to `docs/ARCHITECTURE.md` §4 and prevents accidental committing of raw binary datasets.
- **Consequences:** All paths and commands operate from the root of `Masters_project`.

### 2026-10-16: DEC-002 - Dependency Groups & Tooling Strategy
- **Date:** 2026-10-16
- **Decision:** Configured `pyproject.toml` with `uv` dependency groups: `core`, `ml`, `api`, `dashboard`, and `dev` targeting Python 3.11. Pinned code quality tools to Ruff (100-character line length, banned APIs for system clock), strict Mypy, and Pytest with 80% minimum coverage.
- **Context/Reason:** Fulfills TRD §1, TRD §3, CODE_STYLE §1, and ARCHITECTURE §4 light-image constraints.
- **Consequences:** Heavy ML dependencies are isolated from lightweight API and dashboard runtime contexts.
