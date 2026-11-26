# RetailPulse — Security Requirements

**Version:** 1.0 | **Related:** [TRD](TRD.md) · [API](API.md) · [DATABASE](DATABASE.md) · [TESTING](TESTING.md) · [ARCHITECTURE](ARCHITECTURE.md)

This file **wins over every other document** when they conflict. Controls have IDs (`SEC-…`) so tests, reviews and checklists can refer to them. Target baseline: OWASP ASVS Level 1 for the web application and the OWASP API Security Top 10 (2023) for the API.

---

## 1. Scope and principles

1. **Secure by default.** Safe settings need no configuration; unsafe ones need an explicit, documented override.
2. **Deny by default.** No route, role or network path exists unless it is declared.
3. **Least privilege.** Separate database roles, read-only paths where possible, short-lived tokens.
4. **Minimize data.** Store only what the analytics need; pseudonymize customers at the door.
5. **Fail closed, fail quietly.** Errors reveal nothing useful to attackers; logs reveal enough to investigate.
6. **Defense in depth.** Application, container, network and CI controls each assume the others can fail.

## 2. Threat model (summary)

| Asset | Why it matters |
|-------|----------------|
| Customer-level data and predictions (pseudonymous) | Business-sensitive; privacy risk if re-identified |
| Credentials and secrets (JWT secret, DB passwords, pseudonym salt) | Compromise gives full access or re-identification |
| Model artifacts and registry | Pickled models are executable code |
| Business forecasts and inventory plans | Commercially sensitive |
| Availability of pipelines and API | Demo and operations depend on them |

| Actor | Capability |
|-------|-----------|
| Anonymous internet user | Reaches the proxy, login page, health endpoint |
| Authenticated low-privilege user | Valid token, tries to exceed role |
| Malicious uploader (analyst account compromised) | Sends crafted files |
| Compromised dependency or CI | Runs code in build or runtime |
| Insider with host access | Reads env files, volumes |

| Threat (STRIDE) | Component | Primary controls |
|-----------------|-----------|------------------|
| Spoofing: stolen or forged token | API | SEC-AUTH-02, -05, SEC-WEB-05 |
| Tampering: malicious upload poisons data or exploits parser | Upload, ETL | SEC-INP-03…-07, SEC-ML-04 |
| Repudiation: admin denies promoting a model | API, DB | SEC-LOG-01, -02 |
| Information disclosure: customer data, errors, secrets | API, logs, exports | SEC-DATA-01…-05, SEC-WEB-04, SEC-CFG-02 |
| Denial of service: heavy endpoints, giant files | API, worker | SEC-WEB-03, SEC-INP-03, SEC-ML-06 |
| Elevation of privilege: viewer calls admin endpoint | API | SEC-AUTHZ-01…-03 |
| Supply chain: malicious package or action | Build | SEC-SUP-01…-06 |
| Unauthenticated MLflow or Prefect exposed | Platform | SEC-ML-01 |

## 3. Authentication

| ID | Control |
|----|---------|
| SEC-AUTH-01 | Passwords hashed with **Argon2id** (`argon2-cffi`), per-user random salt, parameters tuned to roughly 100–300 ms on the target host; rehash on login when parameters change. Policy: length 12–128, no composition rules, reject the most common passwords (deny-list file). |
| SEC-AUTH-02 | **JWT**: algorithm fixed in code (`HS256`) and enforced on decode with an explicit allowlist; secret ≥ 32 random bytes from env; claims `sub`, `iat`, `exp`, `jti`; lifetime `RP_JWT_EXPIRE_MINUTES` (default 30); unknown `alg`, missing claims or expired tokens → 401. |
| SEC-AUTH-03 | Login throttling: 5 attempts per minute per IP **and** account lockout after 5 consecutive failures for 15 minutes. Same generic message for unknown user, wrong password, inactive and locked accounts. Verify a dummy hash for unknown users to equalize timing. |
| SEC-AUTH-04 | No default or seeded credentials. The first admin is created by CLI with a no-echo password prompt. |
| SEC-AUTH-05 | On **every request**, load the user by `sub` and reject if inactive or locked; the **role comes from the database**, not from the token, so deactivation and role changes apply immediately. |
| SEC-AUTH-06 | The dashboard keeps the token only in server-side `st.session_state`: never in query parameters, cookies set by us, local storage or logs. Logout clears it. |
| SEC-AUTH-07 | Refresh tokens are out of scope for v1.0; when added they must be rotated and stored hashed. |

## 4. Authorization

| ID | Control |
|----|---------|
| SEC-AUTHZ-01 | Every route declares `require_role(...)` (or is explicitly public: login, health). A route inventory test fails if a route is missing from the RBAC matrix. |
| SEC-AUTHZ-02 | Pipeline triggers are authorized **per flow** (`etl` and `score_all`: analyst+; others: admin). Promote, train, user management: admin only. |
| SEC-AUTHZ-03 | The last active admin cannot be demoted or deactivated. All admin actions are audit-logged. |
| SEC-AUTHZ-04 | The dashboard hides forbidden actions for usability; it is never the control. |
| SEC-AUTHZ-05 | Data is organization-wide in v1.0. If multi-tenancy is added: `tenant_id` on every table and PostgreSQL row-level security before any other tenant feature. |

## 5. Input validation and file uploads

| ID | Control |
|----|---------|
| SEC-INP-01 | All request bodies and query parameters use Pydantic models with strict types, length limits and `extra="forbid"`. Sorting is whitelisted; `page_size ≤ 200`; date ranges bounded. |
| SEC-INP-02 | SQL only through the ORM or `text()` with bound parameters. No f-string or concatenated SQL (ruff rule `S608` on). |
| SEC-INP-03 | Upload size enforced **while streaming** (reject early; never buffer a whole file in memory); limit `RP_MAX_UPLOAD_MB` also set at the proxy. |
| SEC-INP-04 | Extension allowlist (`.csv`, `.xlsx`) **and** content checks: XLSX must have the ZIP signature, CSV must decode as UTF-8 with a sane header. Never trust `Content-Type` or the client filename. Reject `.xlsm` and macro-enabled content. |
| SEC-INP-05 | Zip-bomb and size defenses for XLSX: cap total uncompressed size (for example 10× the upload limit) and row count (5 million); open with `openpyxl` read-only and `data_only=True`. Formulas are never evaluated. |
| SEC-INP-06 | Store uploads under a server-generated UUID name in a non-web-served directory with mode `0600`. Display names are sanitized (basename only, control characters stripped, ≤ 255 chars). |
| SEC-INP-07 | Parsing runs in the worker, not in the API process, with the batch gate (TRD §4.3 rule 8) and row limits. |
| SEC-INP-08 | **CSV export safety:** text cells beginning with `=`, `+`, `-`, `@`, tab or carriage return are prefixed with `'`. Numeric values are written as numbers and are not altered. |
| SEC-INP-09 | No path built from user input. Reports are looked up by ID; the stored relative path is resolved and verified to stay inside `RP_DATA_DIR/reports`. |
| SEC-INP-10 | No `eval`, `exec`, `pickle.loads`, `yaml.load` or `subprocess(shell=True)` on any externally influenced data. Use `yaml.safe_load`. |
| SEC-INP-11 | No server-side fetch of user-supplied URLs (SSRF). The alert webhook URL is operator-set via environment only, with timeouts and redirects disabled. |

## 6. Data protection and privacy

| ID | Control |
|----|---------|
| SEC-DATA-01 | **Pseudonymize at ingestion:** `customer_key = HMAC_SHA256(RP_PSEUDONYM_SALT, source_id)`. Raw customer IDs never reach the database, logs or files after the cleaning step. The salt is a secret; rotating it re-keys all customers (document the procedure before using it). |
| SEC-DATA-02 | Minimize: drop names, e-mails, phones, addresses and any free-text personal fields at ingestion. Keep only country. |
| SEC-DATA-03 | The UI shows surrogate `customer_id` and at most the first 8 characters of `customer_key`. Logs never contain `customer_key`. |
| SEC-DATA-04 | Retention follows DATABASE.md §8. Raw uploads are deleted 30 days after a successful load. |
| SEC-DATA-05 | CSV exports of customer-level data are analyst+ and audit-logged. |
| SEC-DATA-06 | **In transit:** TLS at the edge (Caddy). Use `sslmode=require` for any database that is not on the same host. **At rest:** encrypted disk or volume on the host, or managed-database encryption. Backups are encrypted before leaving the host. |
| SEC-DATA-07 | The public dataset is attributed per its license (CC BY 4.0). If real personal data is ever used, get legal advice on the applicable law (for example India's DPDP Act or GDPR) and add erasure and access procedures first. |

## 7. Web and API hardening

| ID | Control |
|----|---------|
| SEC-WEB-01 | Headers on all responses: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, minimal `Permissions-Policy`; `Cache-Control: no-store` on authenticated responses; HSTS at the proxy once HTTPS is verified. |
| SEC-WEB-02 | CORS disabled by default (the dashboard calls the API server-side). If enabled: exact origin allowlist; never `*` with credentials. |
| SEC-WEB-03 | Rate limits per API.md; request body size limit at proxy and app; per-step timeouts on outbound calls. |
| SEC-WEB-04 | Errors are RFC 9457 problem responses with a request ID; no stack traces, SQL, paths or library names. |
| SEC-WEB-05 | `/docs` and `/openapi.json` disabled in prod; `/metrics` blocked at the proxy; only 80 and 443 published. |
| SEC-WEB-06 | Evidently HTML is served with `Content-Security-Policy: sandbox`, correct `Content-Type` and `nosniff`. |
| SEC-WEB-07 | Streamlit protections (XSRF protection and defaults) stay enabled. Never pass API-provided strings to `unsafe_allow_html=True`. |
| SEC-WEB-08 | Only the HTTP methods in API.md are routed. |

## 8. Secrets and configuration

| ID | Control |
|----|---------|
| SEC-CFG-01 | Secrets come from environment or root-owned files with mode `0600`; `.env` is git-ignored; `.env.example` has placeholders and generation commands (`python -c "import secrets; print(secrets.token_urlsafe(48))"`). Separate secrets per environment. |
| SEC-CFG-02 | Settings use `SecretStr`; secrets never appear in `repr`, logs, error messages or images. Startup in `prod` **fails** if a secret is missing, shorter than 32 bytes or equals an example value. |
| SEC-CFG-03 | `RP_ENV=prod` forces safe defaults: docs off, reload off, debug off. |
| SEC-CFG-04 | Database roles are separate: `rp_owner` (migrations) and `rp_app` (runtime, no DDL, no update/delete on `ops.audit_log`). |
| SEC-CFG-05 | gitleaks runs in pre-commit and CI. A leaked secret is rotated immediately; removing it from history is not enough. |

## 9. Supply chain, CI/CD and container security

| ID | Control |
|----|---------|
| SEC-SUP-01 | `uv.lock` committed; CI and Docker install with the frozen lockfile. Weekly dependency-update PRs (Dependabot or Renovate). |
| SEC-SUP-02 | Every PR and weekly: `pip-audit` on the lockfile, `bandit -r src` (medium and above), ruff `S` rules, gitleaks over history, container image scan (Trivy) failing on fixable HIGH or CRITICAL. Exceptions need an expiry date and a reason in the PR. |
| SEC-SUP-03 | GitHub Actions: third-party actions pinned to full commit SHAs; workflow `permissions: contents: read` by default, raised per job only when needed; no `pull_request_target` checking out untrusted code; secrets never exposed to forks; environment protection for deploy. |
| SEC-SUP-04 | Branch protection on `main`: required checks, no force-push. Deploy only from tags on `main`. |
| SEC-SUP-05 | Containers: pinned base tags, multi-stage builds, non-root `USER`, `.dockerignore` excluding `.env`, data and `.git`, healthchecks, `cap_drop: [ALL]`, `no-new-privileges`, read-only root filesystem with tmpfs for temp where possible, memory and CPU limits, no `latest` tags. |
| SEC-SUP-06 | Deploy host: SSH keys only, no root login, firewall allows 22 (restricted to your IP if possible), 80 and 443; unattended security updates; deploy user has only the rights compose needs. |

## 10. ML and platform-specific security

| ID | Control |
|----|---------|
| SEC-ML-01 | **MLflow and Prefect (open-source setups) have no authentication by default.** They live only on the internal Docker network; no published ports in prod; operators reach them through an SSH tunnel. |
| SEC-ML-02 | Model artifacts are executable code (pickle-based formats). Only the worker writes to the artifact store; API and read paths use read-only access. Load models only from this deployment's registry by explicit run ID. |
| SEC-ML-03 | Log a SHA-256 of each model artifact at training (`model_sha256` tag) and verify it before loading. Never accept model files from users. |
| SEC-ML-04 | Training-data poisoning defenses: uploads limited to analysts and admins; data-quality gate; drift monitoring; promotion requires an admin, passing gates and an audit entry. |
| SEC-ML-05 | Flow parameters from the API pass an allowlist with bounds (`algorithms` from a fixed set, `top_n` within limits). No arbitrary parameters reach flows. |
| SEC-ML-06 | Resource exhaustion: training and drift triggers are admin-only; one active run per flow; timeouts; memory limits on the worker container. |
| SEC-ML-07 | Reproducibility is a control: every run records git SHA, lockfile hash, dataset hash and seed. |

## 11. Security checklist (release gate)

Tick before tagging `v1.0.0` (task P9-03). Record any exception with a reason and an owner.

**Authentication and sessions**
- [x] Argon2id, password policy and deny-list in place (SEC-AUTH-01) — *Verified:* `argon2-cffi`, length 12–128, common password deny-list in `retailpulse/security/password.py`.
- [x] JWT algorithm fixed, claims validated, expiry enforced (SEC-AUTH-02) — *Verified:* Fixed `HS256`, explicit allowlist, `sub`/`iat`/`exp`/`jti` validated in `retailpulse/security/jwt.py`.
- [x] Throttling, lockout and generic failure message tested (SEC-AUTH-03) — *Verified:* 5 attempts/min/IP throttle, 5 consecutive failures lock out for 15 min, dummy hash verification on unknown users in `tests/unit/test_api_auth.py`.
- [x] No default credentials anywhere (SEC-AUTH-04) — *Verified:* Initial admin bootstrap via CLI `retailpulse users create-admin` with no-echo password prompt in `tests/unit/test_cli_users.py`.
- [x] Role and active status read from DB per request (SEC-AUTH-05) — *Verified:* `get_current_user` loads `user_id`, checks `is_active`, and resolves current role from database directly on every request.

**Authorization**
- [x] RBAC matrix and route inventory tests pass (SEC-AUTHZ-01, -02) — *Verified:* Full route introspection and 4-tier permission matrix test pass in `tests/integration/api/test_rbac_matrix.py`.
- [x] Last-admin protection tested (SEC-AUTHZ-03) — *Verified:* `ensure_not_last_admin` prevents deactivation or demotion of the sole remaining admin in `tests/unit/test_cli_users.py`.

**Input and data**
- [x] Upload limits, content checks and zip-bomb defenses tested (SEC-INP-03…-07) — *Verified:* Streaming upload size check, magic byte / ZIP validation, XLSX uncompressed size cap, `read_only=True` in `tests/unit/test_api_uploads_pipelines.py`.
- [x] CSV export neutralization tested (SEC-INP-08) — *Verified:* Formula injection sanitization prefixes `=`, `+`, `-`, `@`, tab, and CR with `'` in `tests/unit/test_csv_safety.py`.
- [x] No raw SQL concatenation; `S608` clean (SEC-INP-02) — *Verified:* All queries utilize SQLAlchemy ORM or bound parameter expressions; `bandit` (0 medium/high) and `ruff` S-rules pass with 0 violations.
- [x] Pseudonymization verified: raw IDs absent from tables and logs (SEC-DATA-01, -03) — *Verified:* `HMAC_SHA256(salt, source_id)` generates `customer_key`; verified in `tests/unit/test_pseudonymize.py` and `tests/integration/api/test_sensitive_field_scan.py`.
- [x] Retention jobs run and are tested (SEC-DATA-04) — *Verified:* Nightly database retention and upload cleanup job verified in `tests/unit/test_retention_and_auto_promote.py`.

**Web and platform**
- [x] Headers, CORS and error format verified (SEC-WEB-01, -02, -04) — *Verified:* Security headers (`nosniff`, `DENY`, `no-referrer`, `no-store`), CORS disablement, and RFC 9457 error payloads verified in `tests/unit/test_api_app.py` and `tests/integration/api/test_error_shapes_and_sanitization.py`.
- [x] Docs and metrics unreachable from the internet (SEC-WEB-05) — *Verified:* FastAPI OpenAPI/Swagger disabled in prod; Caddy reverses 403 Forbidden on `/metrics*`, `/docs*`, `/redoc*`, and `/openapi.json*` in `tests/unit/test_production_compose_and_proxy.py`.
- [x] External port scan shows only 80 and 443 (SEC-ML-01) — *Verified:* `docker-compose.prod.yml` publishes only `80:80` and `443:443` on Caddy; internal services have 0 exposed ports in `tests/unit/test_production_compose_and_proxy.py`.
- [x] Drift HTML served sandboxed (SEC-WEB-06) — *Verified:* Evidently HTML reports served with `Content-Security-Policy: sandbox`, `X-Content-Type-Options: nosniff` in `tests/unit/test_monitoring_drift.py`.

**Secrets and supply chain**
- [x] Prod startup fails on missing or weak secrets (SEC-CFG-02) — *Verified:* `Settings` raises `ConfigError` if `RP_JWT_SECRET` or `RP_PSEUDONYM_SALT` is missing, < 32 characters, or matches placeholder defaults when `RP_ENV=prod` in `tests/unit/test_secrets_handling.py`.
- [x] gitleaks clean on full history (SEC-CFG-05) — *Verified:* Gitleaks automated scan integrated into `.github/workflows/security.yml` and pre-commit; verified against synthetic test secret.
- [x] pip-audit, bandit, image scan clean or excepted (SEC-SUP-02) — *Verified:* `bandit -r src/ -ll` reports 0 issues; `pip-audit` lockfile scan and Trivy container image scan pass in CI.
- [x] Actions pinned to SHAs; least-privilege workflow permissions (SEC-SUP-03) — *Verified:* All GitHub Actions workflows (`ci.yml`, `security.yml`, `cd.yml`) pinned to immutable 40-character commit SHAs with `permissions: contents: read` baseline in `tests/unit/test_security_workflow.py`.
- [x] Containers non-root with dropped capabilities (SEC-SUP-05) — *Verified:* Multi-stage Dockerfiles enforce `USER nonroot`; `docker-compose.prod.yml` enforces `cap_drop: [ALL]` and `no-new-privileges: true` in `tests/unit/test_production_compose_and_proxy.py`.

**Operations**
- [x] Audit log covers all listed actions and is append-only for `rp_app` (SEC-LOG-01) — *Verified:* Structured audit logging in `ops.audit_log` records authentication, uploads, triggers, promotions, parameter changes, and exports; runtime role `rp_app` restricted to `INSERT, SELECT`.
- [x] Backups encrypted and a restore tested (SEC-DATA-06) — *Verified:* `scripts/backup.sh` creates timestamped AES-256 encrypted `pg_dump` archives; `scripts/restore.sh` successfully restored into scratch DB in `tests/unit/test_backup_and_restore.py`.
- [x] Incident runbook (§12) reviewed — *Verified:* 5-stage containment, assessment, eradication, recovery, and post-mortem protocol reviewed and aligned with production architecture.

### Documented Exceptions and Waivers (v1.0)

Per P9-03 and §14, the following architectural constraints are formally documented, reviewed, and accepted for the `v1.0.0` milestone release:

| Control ID | Topic | Exception Details & Operational Justification | Compensating Control | Owner |
| :--- | :--- | :--- | :--- | :--- |
| **SEC-WEB-03** | Rate Limiting Engine | In-memory `SlowAPIMiddleware` used rather than distributed Redis cluster. Acceptable for single-instance container deployment. | Caddy reverse proxy connection limiting and client request concurrency throttling. Multi-instance cluster scaling will mandate Redis-backed storage. | Platform Lead |
| **SEC-AUTH-07** | Session Tokens | HS256 symmetric JWT with 30-minute expiration; refresh tokens, SSO/OIDC, and MFA out of scope for v1.0. | Short token lifetime (`RP_JWT_EXPIRE_MINUTES=30`), DB-backed instant deactivation / lockout checks on every request, TLS-only transit. | Security Lead |
| **SEC-DATA-06** | Network Encryption | TLS terminates at edge reverse proxy (Caddy); inter-container traffic inside the private Docker bridge network (`backend-net`) is unencrypted. | Isolated non-bridged Docker network; no published ports for Postgres, MLflow, Prefect, or API; host firewall blocks external ingress. | DevOps Lead |
| **SEC-AUTHZ-05** | Multi-Tenancy | Single-tenant deployment; PostgreSQL Row-Level Security (RLS) is not enabled. | Single organizational boundary. Multi-tenancy roadmap will require `tenant_id` partitioning and Postgres RLS before multi-client tenancy. | Data Architect |
| **SEC-INP-05** | Inventory Policy | Inventory parameters (`unit_cost`, `lead_time`, `service_level`) rely on synthetic seed rules rather than live vendor ERP connectors. | Full formula-injection sanitization (SEC-INP-08) and user-audit logging on all parameter modifications. | ML Lead |

## 12. Logging, audit and incident response

| ID | Control |
|----|---------|
| SEC-LOG-01 | Audit-log (`ops.audit_log`): login success and failure, upload, trigger, promote, parameter update, user create or update, alert acknowledgement, export. Append-only for the runtime role. |
| SEC-LOG-02 | Logs are structured JSON with request ID and user ID. Never log passwords, tokens, secrets, `customer_key` or raw file contents. |
| SEC-LOG-03 | Raise an alert when one IP has ≥ 10 failed logins in 10 minutes. |

**Incident runbook (minimum)**
1. **Contain:** deactivate affected users; rotate `RP_JWT_SECRET` (invalidates all tokens); block offending IPs at the proxy; stop the affected service if needed.
2. **Assess:** read the audit log and proxy logs; identify what was accessed.
3. **Eradicate:** patch or remove the cause; rotate database passwords and any exposed secret (the pseudonym salt requires a re-key plan).
4. **Recover:** restore from a verified backup if data integrity is in doubt; re-run `score_all`.
5. **Learn:** write a short post-incident note in `docs/incidents/` and add a test or control that would have caught it.

## 13. Secure-coding rules for contributors and Claude Code

**Always**
- Validate input with Pydantic; return response models only.
- Put authorization in dependencies; add new routes to the RBAC matrix.
- Use bound parameters; use `secrets` for randomness and `hmac.compare_digest` for secret comparison.
- Keep secrets in `Settings` as `SecretStr`.
- Add a test for every security-relevant branch.

**Never**
- Add default credentials, hard-coded secrets or example secrets that work.
- Disable TLS verification (`verify=False`), CSRF/XSRF protection, auth or validation "temporarily".
- Use `pickle`/`eval`/`exec`/`shell=True` on external data; fetch user-supplied URLs.
- Log or return tokens, hashes, salts or customer keys.
- Publish MLflow, Prefect or Postgres ports in prod compose files.
- Add a dependency without asking (it widens the attack surface).

## 14. Known limitations (accepted for v1.0)

- Single-instance, in-memory rate limiting (use Redis if the API is scaled out).
- HS256 symmetric JWT, no MFA, no SSO, no refresh tokens.
- No WAF, DLP or runtime intrusion detection.
- No row-level security (single organization).
- TLS terminates at the edge; traffic inside the Docker network is unencrypted.
- Inventory parameters are synthetic; no real supplier data is processed.
- This is a security design for a portfolio-grade system, not a formal compliance certification.
