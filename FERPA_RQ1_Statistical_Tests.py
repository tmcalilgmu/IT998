"""FERPA RQ1: Structured scenario judgment analyses (N=100).
Run from directory containing the original Excel sample file and updated Qualtrics CSV.
Python source exported from the companion Jupyter notebook.
"""


# FERPA Quantitative Paper | RQ1 Statistical Analysis


# 1. Libraries and file paths

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import MonteCarloMethod
from statsmodels.stats.multitest import multipletests
from IPython.display import display

BASE = Path.cwd()  # Put this notebook alongside the two input files.
QUALTRICS_FILE = BASE / "FPEPG_October 26, 2026_06.13.csv"
SAMPLE_FILE = BASE / "FERPA Survey July 24 2026 - N100 - Most Recent.xlsx"
OUTPUT_DIR = BASE / "RQ1_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RESPONSE_OPTIONS = ["Yes", "No", "Depends", "I do not know"]
SCENARIOS = [f"Scenario {i}" for i in range(1, 10)]
RANDOM_SEED = 2026
PERMUTATIONS = 9_999       # Increase for more precise Monte Carlo p-values.
EXACT_RESAMPLES = 19_999  # For selected sparse 2-by-k tables.

for p in (QUALTRICS_FILE, SAMPLE_FILE):
    if not p.is_file():
        raise FileNotFoundError(f"Missing input file: {p.name}. Place it beside the notebook.")
print("Input files found.")


# 2. Match the original N = 100 analysis sample

# Row 2 of the analysis workbook contains question descriptions, not a participant.
sample_raw = pd.read_excel(SAMPLE_FILE, sheet_name=0, skiprows=[1], dtype=str)
sample_raw["Prolific ID"] = sample_raw["Prolific ID"].fillna("").str.strip()
valid_original_q1 = {"True", "False", "Unsure"}
sample_raw["Q1"] = sample_raw["Q1"].fillna("").str.strip()
sample_ids = set(sample_raw.loc[
    sample_raw["Q1"].isin(valid_original_q1) & (sample_raw["Prolific ID"] != ""),
    "Prolific ID"
])

assert len(sample_ids) == 100, (
    f"Expected 100 valid original participant IDs; found {len(sample_ids)}. "
    "Check the original workbook's column headings and inclusion criteria."
)

# Qualtrics exports have two metadata rows after the variable-name row.
qualtrics = pd.read_csv(
    QUALTRICS_FILE, skiprows=[1, 2], dtype=str, keep_default_na=False, low_memory=False
)
qualtrics["Prolific ID"] = qualtrics["Prolific ID"].str.strip()

needed = ["Prolific ID", "Q23"] + SCENARIOS
missing_columns = sorted(set(needed) - set(qualtrics.columns))
if missing_columns:
    raise ValueError(f"Missing expected Qualtrics columns: {missing_columns}")

# Retain ONLY the structured answers needed for RQ1 (not written explanations).
df = qualtrics.loc[qualtrics["Prolific ID"].isin(sample_ids), needed].copy()
assert len(df) == 100 and df["Prolific ID"].nunique() == 100, (
    "Matched sample must contain exactly 100 unique participants. "
    "Check duplicate IDs and source files before proceeding."
)
for scenario in SCENARIOS:
    df[scenario] = df[scenario].str.strip().replace("", np.nan)
    unexpected = set(df[scenario].dropna()) - set(RESPONSE_OPTIONS)
    if unexpected:
        raise ValueError(f"Unexpected responses in {scenario}: {unexpected}")

quality = pd.DataFrame({
    "Scenario": SCENARIOS,
    "Valid judgments": [df[s].notna().sum() for s in SCENARIOS],
    "Missing judgments": [df[s].isna().sum() for s in SCENARIOS],
})
print("Validated original study sample: N =", len(df))
display(quality)



# 3. Test 1: Response frequencies

frequency_rows = []
for scenario in SCENARIOS:
    counts = df[scenario].value_counts(dropna=True).reindex(RESPONSE_OPTIONS, fill_value=0)
    valid_n = int(counts.sum())
    for response, count in counts.items():
        frequency_rows.append({
            "Scenario": scenario,
            "Judgment": response,
            "Count": int(count),
            "Percent": round(100 * count / valid_n, 1) if valid_n else np.nan,
        })
frequencies = pd.DataFrame(frequency_rows)
frequency_counts = frequencies.pivot(index="Scenario", columns="Judgment", values="Count").reindex(columns=RESPONSE_OPTIONS)
frequency_percent = frequencies.pivot(index="Scenario", columns="Judgment", values="Percent").reindex(columns=RESPONSE_OPTIONS)
print("Response counts")
display(frequency_counts)
print("Percentages")
display(frequency_percent)
frequencies.to_csv(OUTPUT_DIR / "RQ1_response_frequencies.csv", index=False)

ax = frequency_percent.plot(kind="barh", stacked=True, figsize=(10, 6), width=0.8)
ax.invert_yaxis()
ax.set_xlabel("Percentage of responses")
ax.set_ylabel("")
ax.set_xlim(0, 100)
ax.legend(title="Judgment", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "RQ1_scenario_response_distributions.png", dpi=160, bbox_inches="tight")
plt.show()


# 4. Test 2: Normalized Shannon entropy

def normalized_shannon_entropy(counts, number_of_options=4):
    counts = np.asarray(counts, dtype=float)
    if counts.sum() == 0:
        return np.nan
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(number_of_options))

entropy = pd.DataFrame({
    "Scenario": SCENARIOS,
    "Normalized Shannon entropy": [
        normalized_shannon_entropy(frequency_counts.loc[s, RESPONSE_OPTIONS].to_numpy())
        for s in SCENARIOS
    ],
}).sort_values("Normalized Shannon entropy", ascending=False)
display(entropy.round(3))
entropy.to_csv(OUTPUT_DIR / "RQ1_shannon_entropy.csv", index=False)

ax = entropy.sort_values("Normalized Shannon entropy").plot.barh(
    x="Scenario", y="Normalized Shannon entropy", figsize=(8, 5), legend=False
)
ax.set_xlim(0, 1)
ax.set_xlabel("Normalized entropy (0 = complete agreement, 1 = maximum dispersion)")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "RQ1_entropy.png", dpi=160, bbox_inches="tight")
plt.show()


# 5. Define the independent groups for scenario comparisons

def normalize_institution_text(x):
    if pd.isna(x):
        return ""
    # Converts special Unicode spaces, dashes, etc. to comparable text.
    return " ".join(unicodedata.normalize("NFKC", str(x)).lower().split())

def institution_group(value):
    x = normalize_institution_text(value)
    if "kindergarten" in x or "elementary" in x or "middle school" in x or "high school" in x:
        return "K-12"
    if any(word in x for word in ("college", "university", "graduate", "professional school")):
        return "Higher education"
    return np.nan

df["ComparisonGroup"] = df["Q23"].map(institution_group)
GROUP_LABEL = "Survey-reported institution type (Q23)"

SECTOR_OVERRIDE = BASE / "rq1_verified_sectors.csv"
if SECTOR_OVERRIDE.exists():
    roster = pd.read_csv(SECTOR_OVERRIDE, dtype=str)
    if not {"Prolific ID", "RecruitmentSector"}.issubset(roster.columns):
        raise ValueError("Sector roster must contain Prolific ID and RecruitmentSector")
    roster["Prolific ID"] = roster["Prolific ID"].str.strip()
    if roster["Prolific ID"].duplicated().any():
        raise ValueError("Duplicate IDs found in sector roster")
    mapping = roster.set_index("Prolific ID")["RecruitmentSector"].to_dict()
    updated = df["Prolific ID"].map(mapping)
    if updated.isna().any() or not updated.isin(["K-12", "Higher education"]).all():
        raise ValueError("The verified sector roster must classify all 100 study participants")
    df["ComparisonGroup"] = updated
    GROUP_LABEL = "Verified original recruitment sector"

print("Comparison grouping:", GROUP_LABEL)
print(df["ComparisonGroup"].value_counts(dropna=False).to_string())


# 6. Test 3: Permutation Chi-Square

def chi_square_from_counts(table):
    table = np.asarray(table, dtype=float)
    # Entirely unused response categories contribute no information.
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan, table
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    if np.any(expected == 0):
        return np.nan, np.nan, table
    chi2 = float(np.sum((table - expected)**2 / expected))
    return chi2, expected, table

def permutation_chi_square(scenario, group_col="ComparisonGroup", n_resamples=9_999, seed=2026):
    sub = df[[scenario, group_col]].dropna().copy()
    sub = sub.loc[sub[group_col].isin(["K-12", "Higher education"])]
    group_codes = pd.Categorical(sub[group_col], categories=["K-12", "Higher education"]).codes
    judgment_codes = pd.Categorical(sub[scenario], categories=RESPONSE_OPTIONS).codes
    observed_full = np.bincount(group_codes*4 + judgment_codes, minlength=8).reshape(2, 4)
    chi2_obs, expected, observed_used = chi_square_from_counts(observed_full)
    if np.isnan(chi2_obs):
        return {"Chi-square": np.nan, "Permutation p": np.nan, "N compared": len(sub)}

    rng = np.random.default_rng(seed)
    ge = 0
    active_columns = observed_full.sum(axis=0) > 0
    for _ in range(n_resamples):
        shuffled_groups = rng.permutation(group_codes)
        permuted = np.bincount(shuffled_groups*4 + judgment_codes, minlength=8).reshape(2, 4)
        stat, _, _ = chi_square_from_counts(permuted[:, active_columns])
        ge += (stat >= chi2_obs - 1e-12)
    # +1 correction prevents a zero estimated p-value.
    p_value = (ge + 1) / (n_resamples + 1)
    return {
        "Chi-square": chi2_obs,
        "Permutation p": p_value,
        "N compared": len(sub),
    }

comparison_rows = []
for i, scenario in enumerate(SCENARIOS):
    result = permutation_chi_square(scenario, n_resamples=PERMUTATIONS,
                                    seed=RANDOM_SEED + i)
    comparison_rows.append({"Scenario": scenario, **result})
comparisons = pd.DataFrame(comparison_rows)
display(comparisons.round(4))


# 7. Test 4: Cramér's V

def cramers_v(table):
    chi2, _, used = chi_square_from_counts(table)
    n = used.sum()
    denominator = n * min(used.shape[0] - 1, used.shape[1] - 1)
    return np.sqrt(chi2 / denominator) if denominator > 0 else np.nan

def judgment_table(scenario):
    sub = df.loc[df["ComparisonGroup"].isin(["K-12", "Higher education"]),
                 ["ComparisonGroup", scenario]].dropna()
    return pd.crosstab(sub["ComparisonGroup"], sub[scenario]).reindex(
        index=["K-12", "Higher education"],
        columns=RESPONSE_OPTIONS, fill_value=0
    )

comparisons["Cramer's V"] = [cramers_v(judgment_table(s).to_numpy()) for s in SCENARIOS]
display(comparisons.round(4))


# 8. Test 5: Fisher-Freeman-Halton exact tests (selected sparse tables)

from scipy import __version__ as scipy_version
from packaging.version import Version
if Version(scipy_version) < Version("1.15"):
    raise RuntimeError("SciPy 1.15+ required for Monte Carlo Fisher-Freeman-Halton tests")

exact_rows = []
for i, scenario in enumerate(SCENARIOS):
    full = judgment_table(scenario).to_numpy()
    table = full[:, full.sum(axis=0) > 0]
    if table.shape[1] < 3:
        exact_rows.append({"Scenario": scenario, "Reason": "Only two observed response categories",
                           "FFH Monte Carlo p": np.nan})
        continue
    _, expected, _ = chi_square_from_counts(table)
    if expected is None or np.isnan(expected).any() or not (expected < 5).any():
        exact_rows.append({"Scenario": scenario, "Reason": "No expected count < 5",
                           "FFH Monte Carlo p": np.nan})
        continue
    mc = MonteCarloMethod(n_resamples=EXACT_RESAMPLES,
                          rng=np.random.default_rng(RANDOM_SEED + 100 + i))
    result = stats.fisher_exact(table, method=mc)
    exact_rows.append({"Scenario": scenario, "Reason": "Sparse 2-by-k table",
                       "FFH Monte Carlo p": float(result.pvalue)})
exact = pd.DataFrame(exact_rows)
display(exact.round(4))
exact.to_csv(OUTPUT_DIR / "RQ1_exact_test_sensitivity.csv", index=False)


# 9. Test 6: Holm correction for the nine group comparisons

pvals = comparisons["Permutation p"].to_numpy(dtype=float)
valid = np.isfinite(pvals)
comparisons["Holm adjusted p"] = np.nan
comparisons["Significant after Holm (0.05)"] = False
if valid.any():
    reject, adjusted, _, _ = multipletests(pvals[valid], alpha=0.05, method="holm")
    comparisons.loc[valid, "Holm adjusted p"] = adjusted
    comparisons.loc[valid, "Significant after Holm (0.05)"] = reject

comparisons = comparisons.merge(exact[["Scenario", "FFH Monte Carlo p"]],
                                on="Scenario", how="left")
comparisons = comparisons[["Scenario", "N compared", "Chi-square", "Permutation p",
                           "Cramer's V", "Holm adjusted p",
                           "Significant after Holm (0.05)", "FFH Monte Carlo p"]]
display(comparisons.round(4))
comparisons.to_csv(OUTPUT_DIR / "RQ1_group_comparisons_and_holm.csv", index=False)
print("All RQ1 outputs saved to:", OUTPUT_DIR.resolve())


# 10. Quick summary and interpretation notes

# A minimal, reader-friendly RQ1 summary created from the analyses above.
scenario_by_entropy = entropy.iloc[0]
print(f"Highest response dispersion: {scenario_by_entropy['Scenario']} "
      f"(normalized entropy = {scenario_by_entropy['Normalized Shannon entropy']:.3f})")
print("Most common response by scenario:")
for scenario in SCENARIOS:
    row = frequency_counts.loc[scenario]
    choice = row.idxmax()
    print(f"  {scenario}: {choice} ({100 * row.max() / row.sum():.0f}%)")
print(f"Group comparisons remaining significant after Holm: "
      f"{comparisons['Significant after Holm (0.05)'].sum()}")
print("Group definition:", GROUP_LABEL)
