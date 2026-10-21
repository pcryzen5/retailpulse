#!/usr/bin/env python3
"""Generate a deterministic test fixture (mini_retail.csv) for RetailPulse ETL testing.

Creates tests/fixtures/mini_retail.csv (approx 3,000 rows) covering all edge cases
mandated by TRD §4.3 and P1-01:
  - Valid sales spanning complete trading days
  - Cancellations/returns (Invoice starts with C, negative quantity)
  - Non-product SKUs (POST, DOT, M, BANK CHARGES)
  - Real-looking DCGS* product lines (e.g., DCGS0003, party bags)
  - Null/missing customer IDs
  - Exact duplicate rows
  - Simulated cross-sheet overlap rows in 2010-12-01 to 2010-12-09
  - Mixed number/text codes (StockCode as pure digits and alphanumeric)
  - Case-variant SKUs (e.g., 85123a vs 85123A)
  - Outlier order (>= 50,000 quantity) and cancellation
  - Invalid cancellation with positive quantity (C-invoice with positive quantity)
  - Partial last day (2011-12-09 ending at 12:50)
  - Bad/coercion-failure rows (malformed prices or dates)
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
TARGET_PATH = Path("tests/fixtures/mini_retail.csv")

# Representative catalog of products and SKUs
PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.95),
    ("71053", "WHITE METAL LANTERN", 3.39),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.75),
    ("84029E", "RED WOOLLY HOTTIE WHITE HEART.", 3.39),
    ("22752", "SET 7 BABUSHKA NESTING BOXES", 7.65),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
    ("22632", "HAND WARMER RED RETROSPOT", 1.85),
    ("22633", "HAND WARMER UNION JACK", 1.85),
    ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
    ("21212", "PACK OF 72 RETRO SPOT CAKE CASES", 0.55),
    ("21977", "PACK OF 60 PINK PAW CAKE CASES", 0.55),
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
    ("22457", "NATURAL SLATE HEART CHALKBOARD", 2.95),
    ("47566", "PARTY BUNTING", 4.65),
    ("85099B", "JUMBO BAG RED RETROSPOT", 1.95),
    ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
    ("20727", "LUNCH BAG BLACK SKULL.", 1.65),
    ("20728", "LUNCH BAG CARS BLUE", 1.65),
    ("22383", "LUNCH BAG SUKI  DESIGN", 1.65),
    ("DCGS0003", "BOXED GLASS ASHTRAY", 2.55),  # Real product with DCGS prefix
    ("DCGSSGIRL", "GIRLS PARTY BAG", 3.20),  # Real product with DCGS prefix
]

NON_PRODUCT_SKUS = [
    ("POST", "POSTAGE", 18.00),
    ("DOT", "DOTCOM POSTAGE", 42.50),
    ("M", "Manual", 5.00),
    ("BANK CHARGES", "Bank Charges", 15.00),
    ("C2", "CARRIAGE", 50.00),
]

COUNTRIES = [
    "United Kingdom",
    "United Kingdom",
    "United Kingdom",
    "Germany",
    "France",
    "EIRE",
    "Spain",
    "Netherlands",
]


def generate_fixture(output_path: Path = TARGET_PATH) -> None:
    """Generate deterministic mini_retail.csv dataset."""
    rng = random.Random(SEED)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[list[str]] = []
    headers = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]

    invoice_counter = 500000
    customer_ids = [f"{12000 + i}" for i in range(1, 151)]

    # 1. Base realistic sales lines across dates (2010-10-01 to 2011-12-08)
    start_date = datetime(2010, 10, 1, 9, 0, 0)
    current_date = start_date

    while len(rows) < 2700:
        # Skip Saturdays (TRD §4.4)
        if current_date.weekday() == 5:
            current_date += timedelta(days=1)
            continue

        num_orders = rng.randint(4, 10)
        for _ in range(num_orders):
            invoice_counter += 1
            invoice_no = str(invoice_counter)
            country = rng.choice(COUNTRIES)
            customer = rng.choice(customer_ids) if rng.random() > 0.20 else ""
            order_time = current_date + timedelta(
                hours=rng.randint(0, 8),
                minutes=rng.randint(0, 59),
                seconds=rng.randint(0, 59),
            )
            time_str = order_time.strftime("%Y-%m-%d %H:%M:%S")

            # Lines per order
            lines_in_order = rng.randint(1, 4)
            for _ in range(lines_in_order):
                prod_sku, desc, price = rng.choice(PRODUCTS)
                qty = rng.choice([1, 2, 3, 6, 12, 24])

                # Float representation of customer ID often found in original XLSX (e.g. 12345.0)
                cust_repr = f"{customer}.0" if customer and rng.random() > 0.5 else customer

                rows.append(
                    [
                        invoice_no,
                        prod_sku,
                        desc,
                        str(qty),
                        time_str,
                        f"{price:.2f}",
                        cust_repr,
                        country,
                    ]
                )

        current_date += timedelta(days=1)

    # 2. Cancellations / Returns (valid: C prefix, negative quantity)
    for _ in range(50):
        c_inv = f"C{rng.randint(510000, 530000)}"
        prod_sku, desc, price = rng.choice(PRODUCTS)
        qty = -rng.randint(1, 12)
        cust = rng.choice(customer_ids)
        order_time = start_date + timedelta(days=rng.randint(1, 100), hours=rng.randint(9, 17))
        rows.append(
            [
                c_inv,
                prod_sku,
                f"Returned {desc}",
                str(qty),
                order_time.strftime("%Y-%m-%d %H:%M:%S"),
                f"{price:.2f}",
                cust,
                "United Kingdom",
            ]
        )

    # 3. Non-product SKUs (POST, DOT, M, BANK CHARGES)
    for _ in range(40):
        inv = str(rng.randint(520000, 540000))
        sku, desc, price = rng.choice(NON_PRODUCT_SKUS)
        order_time = start_date + timedelta(days=rng.randint(1, 100), hours=rng.randint(9, 17))
        cust = rng.choice(customer_ids) if rng.random() > 0.1 else ""
        rows.append(
            [
                inv,
                sku,
                desc,
                "1",
                order_time.strftime("%Y-%m-%d %H:%M:%S"),
                f"{price:.2f}",
                cust,
                "United Kingdom",
            ]
        )

    # 4. Mixed number/text codes and case-variant SKUs
    # 85123a (lowercase variant)
    rows.append(
        [
            "530001",
            "85123a",
            "WHITE HANGING HEART T-LIGHT HOLDER (LOWERCASE)",
            "6",
            "2010-11-15 11:20:00",
            "2.95",
            "12345",
            "United Kingdom",
        ]
    )
    # Pure numeric SKU
    rows.append(
        [
            "530002",
            "21232",
            "RED SPOTTY BISCUIT TIN",
            "4",
            "2010-11-15 11:22:00",
            "3.75",
            "12346",
            "United Kingdom",
        ]
    )

    # 5. Outlier order (>= 50,000 units) and corresponding cancellation
    outlier_date = "2011-01-18 10:01:00"
    rows.append(
        [
            "541431",
            "23166",
            "MEDIUM CERAMIC TOP STORAGE JAR",
            "74215",
            outlier_date,
            "1.04",
            "12346",
            "United Kingdom",
        ]
    )
    rows.append(
        [
            "C541433",
            "23166",
            "MEDIUM CERAMIC TOP STORAGE JAR",
            "-74215",
            "2011-01-18 10:17:00",
            "1.04",
            "12346",
            "United Kingdom",
        ]
    )

    # 6. Invalid cancellation with positive quantity (Rule 4: rejected line)
    rows.append(
        [
            "C539999",
            "22752",
            "SET 7 BABUSHKA NESTING BOXES",
            "2",  # Positive quantity on a C invoice is invalid!
            "2010-12-05 14:10:00",
            "7.65",
            "12347",
            "United Kingdom",
        ]
    )

    # 7. Cross-sheet overlap duplicates (2010-12-01 to 2010-12-09)
    overlap_rows = [
        [
            "536365",
            "85123A",
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "6",
            "2010-12-01 08:26:00",
            "2.55",
            "17850",
            "United Kingdom",
        ],
        [
            "536365",
            "71053",
            "WHITE METAL LANTERN",
            "6",
            "2010-12-01 08:26:00",
            "3.39",
            "17850",
            "United Kingdom",
        ],
        [
            "536366",
            "22633",
            "HAND WARMER UNION JACK",
            "6",
            "2010-12-01 08:28:00",
            "1.85",
            "17850",
            "United Kingdom",
        ],
    ]
    # Add sheet 1 copy and sheet 2 copy (identical)
    rows.extend(overlap_rows)
    rows.extend(overlap_rows)

    # 8. Exact duplicate rows within dataset
    dup_row = [
        "538000",
        "22423",
        "REGENCY CAKESTAND 3 TIER",
        "1",
        "2010-12-08 15:30:00",
        "12.75",
        "14000",
        "United Kingdom",
    ]
    rows.append(dup_row)
    rows.append(dup_row)

    # 9. Partial last day (2011-12-09 ending early at 12:50)
    rows.append(
        [
            "581587",
            "22632",
            "HAND WARMER RED RETROSPOT",
            "12",
            "2011-12-09 12:50:00",
            "1.85",
            "12680",
            "France",
        ]
    )

    # 10. Bad / coercion-failure rows (quarantined)
    rows.append(
        [
            "590001",
            "22720",
            "SET OF 3 CAKE TINS PANTRY DESIGN",
            "3",
            "INVALID-DATE-TIME",  # Unparseable date
            "4.95",
            "12348",
            "United Kingdom",
        ]
    )
    rows.append(
        [
            "590002",
            "21212",
            "PACK OF 72 RETRO SPOT CAKE CASES",
            "10",
            "2011-05-10 14:00:00",
            "FREE",  # Non-numeric price
            "12349",
            "United Kingdom",
        ]
    )

    # Write CSV with newline='\n' for exact byte reproducibility
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    generate_fixture()
