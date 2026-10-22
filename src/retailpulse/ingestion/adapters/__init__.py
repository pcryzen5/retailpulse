"""Ingestion source adapters for Online Retail II and Generic CSV."""

from retailpulse.ingestion.adapters.generic_csv import GenericCSVAdapter
from retailpulse.ingestion.adapters.online_retail_ii import OnlineRetailIIAdapter

__all__ = ["GenericCSVAdapter", "OnlineRetailIIAdapter"]
