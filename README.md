# RetailPulse

RetailPulse is an AI-powered customer analytics and demand forecasting platform for retail businesses. It provides:
1. **Demand forecasting** per SKU with calibrated uncertainty intervals.
2. **Customer segmentation** using RFM-based clustering.
3. **Churn prediction** with interpretable explanations (SHAP).
4. **Inventory optimization** via $(s, Q)$ policy recommendations and simulation.

Served through a **FastAPI** REST API and an interactive **Streamlit** dashboard, orchestrated by **Prefect**, tracked by **MLflow**, and stored in **PostgreSQL 16**.

## Project Status

Phase 0: Foundations & Environment Setup in progress.
For detailed delivery progress, see [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) and [docs/PHASES.md](docs/PHASES.md).

## Quickstart

```bash
# Setup virtual environment and pre-commit hooks
make setup

# Start backing services (PostgreSQL, MLflow, Prefect)
make up

# Run checks
make lint typecheck test
```
