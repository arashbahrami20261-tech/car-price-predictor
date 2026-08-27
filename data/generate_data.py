"""Generate a synthetic used-car dataset for the Iranian second-hand market.

The data is generated, not scraped, so the repository stays self-contained and
reproducible. Messiness is injected on purpose (missing values, inconsistent
category spelling, outliers) so that the training pipeline has to handle the
same problems it would face with real scraped listings.

Run:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 4000
CURRENT_YEAR = 2026

# base_price_million_toman refers to a mint, zero-mileage car of that model.
CAR_CATALOG = {
    "Pride": {"base": 320, "engine": 1.3, "decay": 0.055},
    "Tiba": {"base": 430, "engine": 1.5, "decay": 0.055},
    "Quik": {"base": 520, "engine": 1.5, "decay": 0.050},
    "Peugeot 206": {"base": 720, "engine": 1.4, "decay": 0.050},
    "Peugeot Pars": {"base": 780, "engine": 1.8, "decay": 0.048},
    "Samand": {"base": 640, "engine": 1.8, "decay": 0.052},
    "Dena": {"base": 980, "engine": 1.7, "decay": 0.045},
    "Shahin": {"base": 1050, "engine": 1.5, "decay": 0.045},
    "Rana": {"base": 760, "engine": 1.6, "decay": 0.048},
    "Hyundai Accent": {"base": 2400, "engine": 1.6, "decay": 0.038},
    "Toyota Corolla": {"base": 3100, "engine": 1.8, "decay": 0.032},
    "Kia Cerato": {"base": 2700, "engine": 2.0, "decay": 0.035},
}

FUEL_TYPES = ["petrol", "petrol_lpg"]
TRANSMISSIONS = ["manual", "automatic"]
BODY_CONDITIONS = ["intact", "minor_paint", "major_paint", "accident_repaired"]
COLORS = ["white", "silver", "black", "blue", "grey", "red"]

# Multipliers applied to the base price.
CONDITION_MULTIPLIER = {
    "intact": 1.00,
    "minor_paint": 0.92,
    "major_paint": 0.80,
    "accident_repaired": 0.66,
}
COLOR_MULTIPLIER = {
    "white": 1.03,
    "silver": 1.00,
    "black": 1.01,
    "blue": 0.97,
    "grey": 0.98,
    "red": 0.95,
}
TRANSMISSION_MULTIPLIER = {"manual": 1.00, "automatic": 1.12}
FUEL_MULTIPLIER = {"petrol": 1.00, "petrol_lpg": 0.95}


def build_dataframe(rng: np.random.Generator) -> pd.DataFrame:
    """Create the clean signal before any noise or damage is applied."""
    models = rng.choice(list(CAR_CATALOG), size=N_ROWS)
    years = rng.integers(2005, CURRENT_YEAR + 1, size=N_ROWS)
    age = CURRENT_YEAR - years

    # Older cars have driven more, with wide spread around 18k km per year.
    mileage = rng.normal(loc=age * 18000 + 5000, scale=25000)
    mileage = np.clip(mileage, 0, 700000).round(-2)

    conditions = rng.choice(BODY_CONDITIONS, size=N_ROWS, p=[0.45, 0.28, 0.17, 0.10])
    transmissions = rng.choice(TRANSMISSIONS, size=N_ROWS, p=[0.72, 0.28])
    fuels = rng.choice(FUEL_TYPES, size=N_ROWS, p=[0.78, 0.22])
    colors = rng.choice(COLORS, size=N_ROWS, p=[0.30, 0.22, 0.18, 0.11, 0.11, 0.08])
    owners = rng.integers(1, 6, size=N_ROWS)

    base = np.array([CAR_CATALOG[m]["base"] for m in models])
    decay = np.array([CAR_CATALOG[m]["decay"] for m in models])
    engine = np.array([CAR_CATALOG[m]["engine"] for m in models])

    price = base * np.exp(-decay * age)
    price *= 1 - np.clip(mileage / 1_000_000, 0, 0.45)
    price *= np.array([CONDITION_MULTIPLIER[c] for c in conditions])
    price *= np.array([COLOR_MULTIPLIER[c] for c in colors])
    price *= np.array([TRANSMISSION_MULTIPLIER[t] for t in transmissions])
    price *= np.array([FUEL_MULTIPLIER[f] for f in fuels])
    price *= 1 - (owners - 1) * 0.025
    price *= rng.normal(1.0, 0.07, size=N_ROWS)  # irreducible market noise
    price = np.clip(price, 60, None).round(1)

    return pd.DataFrame(
        {
            "model": models,
            "year": years,
            "mileage_km": mileage.astype(int),
            "engine_size_l": engine,
            "fuel_type": fuels,
            "transmission": transmissions,
            "body_condition": conditions,
            "color": colors,
            "previous_owners": owners,
            "price_million_toman": price,
        }
    )


def dirty_dataframe(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject the kind of damage found in real scraped listings."""
    df = df.copy()

    # 1. Missing values in two columns.
    for column, rate in (("mileage_km", 0.06), ("body_condition", 0.04)):
        mask = rng.random(len(df)) < rate
        df.loc[mask, column] = np.nan

    # 2. Inconsistent category spelling and casing.
    mask = rng.random(len(df)) < 0.08
    df.loc[mask, "transmission"] = df.loc[mask, "transmission"].str.upper()
    mask = rng.random(len(df)) < 0.06
    df.loc[mask, "color"] = " " + df.loc[mask, "color"] + " "

    # 3. Impossible mileage values (data-entry errors: km typed as metres).
    mask = rng.random(len(df)) < 0.015
    df.loc[mask, "mileage_km"] = df.loc[mask, "mileage_km"] * 1000

    # 4. Exact duplicate rows.
    duplicates = df.sample(frac=0.02, random_state=RANDOM_SEED)
    df = pd.concat([df, duplicates], ignore_index=True)

    return df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    df = dirty_dataframe(build_dataframe(rng), rng)
    df.to_csv("data/cars.csv", index=False)
    print(f"Wrote data/cars.csv with {len(df)} rows and {df.shape[1]} columns.")


if __name__ == "__main__":
    main()
