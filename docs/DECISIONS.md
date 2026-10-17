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

### 2026-10-16: DEC-003 - Package Skeleton, Error Hierarchy, and Clock Isolation
- **Date:** 2026-10-16
- **Decision:** Implemented `src/retailpulse/{config,logging,errors,timeutil}.py`. `config.Settings` enforces all TRD §3 parameters, fails on prod if secrets < 32 bytes or placeholders. `errors.py` defines domain exception taxonomy mapped to RFC 9457. `logging.py` standardizes structlog JSON output with request_id contextvar. `timeutil.utc_now()` is designated as the sole system clock interface.
- **Context/Reason:** Establishes predictable domain error handling, structured telemetry, and prevents temporal data leakage across ETL and ML components.
- **Consequences:** All subsequent domain modules inherit from `RetailPulseError` and accept explicit temporal cutoffs (`as_of`) instead of querying the system clock.

### 2026-10-17: DEC-004 - Unified Makefile and Package Boundary Enforcement
- **Date:** 2026-10-17
- **Decision:** Created cross-platform `Makefile` supporting all 21 targets from Appendix A (`help`, `setup`, `lint`, `fmt`, `typecheck`, `test`, etc.) with dynamic self-documentation. Configured subpackage stubs across all architectural layers with `import-linter` contract verification.
- **Context/Reason:** Fulfills Appendix A and enforces module dependency rules from `docs/ARCHITECTURE.md` §4 during CI and local development.
- **Consequences:** Standardized developer workflow across environments with automated verification of layer isolation and temporal clock boundaries.
