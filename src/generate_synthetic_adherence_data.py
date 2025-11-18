import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 4000  # number of patients

def generate_adherence_data(n_rows: int = N) -> pd.DataFrame:
    patient_id = np.arange(1, n_rows + 1)

    age = np.random.randint(18, 90, size=n_rows)
    gender = np.random.choice(["M", "F"], size=n_rows)

    # Chronic condition counts
    chronic_conditions = np.random.choice(
        [0, 1, 2, 3, 4],
        size=n_rows,
        p=[0.15, 0.30, 0.25, 0.20, 0.10]
    )

    # Number of active meds
    num_meds = np.random.choice(
        [1, 2, 3, 4, 5, 6, 7, 8],
        size=n_rows,
        p=[0.05, 0.10, 0.20, 0.25, 0.20, 0.10, 0.06, 0.04]
    )

    # Prior year refill gap (days without medication on hand)
    refill_gap_days = np.clip(
        np.random.normal(loc=20, scale=15, size=n_rows),
        a_min=0,
        a_max=120
    ).astype(int)

    # Prior year adherence percentage (0–100)
    prior_year_adherence = np.clip(
        100 - refill_gap_days + np.random.normal(0, 8, size=n_rows),
        a_min=0,
        a_max=100
    )

    # Has depression/anxiety diagnosis (yes/no)
    mental_health_flag = np.random.choice([0, 1], size=n_rows, p=[0.7, 0.3])

    # Copay level
    copay_tier = np.random.choice(
        ["low", "medium", "high"],
        size=n_rows,
        p=[0.45, 0.35, 0.20]
    )

    # Plan type
    plan_type = np.random.choice(
        ["Commercial", "Medicare", "Medicaid"],
        size=n_rows,
        p=[0.55, 0.30, 0.15]
    )

    # Base log-odds for being adherent
    log_odds = (
        -1.0
        + 0.03 * (age - 50)              # older slightly more adherent
        - 0.25 * chronic_conditions      # more conditions → more complexity
        - 0.12 * num_meds                # more meds → harder to stay adherent
        - 0.03 * refill_gap_days         # bigger gaps → lower adherence
        + 0.04 * (prior_year_adherence - 60)
        - 0.5 * mental_health_flag       # MH comorbidities → more risk
    )

    # Copay impact
    copay_effect = np.where(copay_tier == "low", 0.4,
                     np.where(copay_tier == "medium", 0.0, -0.5))
    log_odds += copay_effect

    # Plan type impact
    plan_effect = np.where(plan_type == "Medicare", 0.2,
                    np.where(plan_type == "Medicaid", -0.2, 0.0))
    log_odds += plan_effect

    # Convert to probabilities via logistic function
    prob_adherent = 1 / (1 + np.exp(-log_odds))

    # Draw binary outcome: 1 = adherent, 0 = non-adherent
    adherent = np.random.binomial(1, prob_adherent, size=n_rows)

    df = pd.DataFrame({
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "chronic_conditions": chronic_conditions,
        "num_meds": num_meds,
        "refill_gap_days": refill_gap_days,
        "prior_year_adherence": np.round(prior_year_adherence, 1),
        "mental_health_flag": mental_health_flag,
        "copay_tier": copay_tier,
        "plan_type": plan_type,
        "adherent": adherent,
    })

    return df


if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)
    df = generate_adherence_data()
    out_path = "../data/med_adherence_synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
