# Plan: Swap match/mismatch coding convention

**Current**: `y = 0` (match, same species) / `y = 1` (mismatch, different species)  
**Proposed**: `y = 1` (match) / `y = 0` (mismatch)

## Motivation

The current convention (`y = 0` for match) is unintuitive — a "match" is a positive ecological event, but is coded as the zero class. This inverts the natural reading of classification metrics (AUC, precision, recall) and forces workarounds like `1 - p_match` in the AUC calculation. Swapping to `y = 1` (match) aligns with the standard convention that the positive class is the event of interest.

## Impact summary

| Category | Files affected | Locations | Risk |
|----------|---------------|-----------|------|
| Source of truth (y creation) | 2 R files | 6 locations | HIGH — defines everything downstream |
| Loss formula (BCE) | 1 Python file | 2 locations | HIGH — silent sign flip if missed |
| Retrospective correction (r0/r1) | 3 files (1 R, 1 R export, 1 Python) | ~15 locations | HIGH — retention rates semantically tied to y |
| Hard-pair mining | 1 Python file | 4 locations | MEDIUM — cell partitioning + hardness formula |
| AUC / confusion matrix | 2 files (1 Python, 1 R) | 5 locations | MEDIUM — positive-class definition |
| Diagnostic scripts | 3 Python scripts | ~10 locations | LOW — labels and filters |
| Documentation (main.tex) | 1 file | ~8 locations | LOW — equations + narrative |
| CLESSO v2 R pipeline | 6 R files | ~40 locations | LOW — legacy, but must stay consistent |
| Tests | 2 files | 3 locations | LOW |
| Config comments | 2 files | 3 locations | LOW |

**Total: ~18 files, ~95+ individual locations.**

---

## Phase 1: Source of truth — y column creation (R)

### 1.1 `src/shared/R/obs_pair_sampler.R` — lines 108–109, 205–206, 348–349

```r
# CURRENT (3 locations, identical pattern):
match <- frog.auGrid$species[samp1] == frog.auGrid$species[samp2]
obsMatch$Match[place] <- as.numeric(!match)

# PROPOSED:
match <- frog.auGrid$species[samp1] == frog.auGrid$species[samp2]
obsMatch$Match[place] <- as.numeric(match)   # remove negation
```

**Change**: Remove the `!` negation operator in all 3 locations.

### 1.2 `OLD_RECA/code/obsPairSampler-bigData-RECA.r` — lines 66–67, 180–181, 324–325

Same pattern — remove `!` in all 3 locations.  
*Note: this is legacy code; change only if we want consistency for reproducibility.*

---

## Phase 2: Retention rates — semantic swap of r0 ↔ r1

The retention rates `r_0` and `r_1` are **defined by their y-value**: `r_0 = P(retain | y=0)`, `r_1 = P(retain | y=1)`. After the swap, the *match* retention rate becomes `r_1` (since matches are now y=1), and the *mismatch* rate becomes `r_0`.

**Two strategies — pick one:**

- **Option A (rename r0/r1):** Keep r_0 = retention for y=0, r_1 = retention for y=1 — the *meaning* of r_0 flips from "match retention" to "mismatch retention". No formula changes needed, but every *comment* describing r_0 as "match retention" must be updated. *Risk: confusing if someone reads old code comments.*

- **Option B (swap semantic names):** Introduce `r_match` / `r_mismatch` naming. More invasive (touches metadata.json keys, model.py dict keys, export_for_nn.R) but permanently disambiguates. *Recommended.*

### 2.1 `src/clesso_v2/clesso_sampler_optimised.R`

**Within-site retention (lines 331–340)**:
```r
# CURRENT:
r_W_0 <- if (total_drawn_match > 0) n_match / total_drawn_match else 1.0
r_W_1 <- if (total_drawn_mismatch > 0) n_miss / total_drawn_mismatch else 1.0
attr(out, "retention_rates") <- list(r_0 = r_W_0, r_1 = r_W_1, ...)

# PROPOSED (Option B):
r_match    <- if (total_drawn_match > 0) n_match / total_drawn_match else 1.0
r_mismatch <- if (total_drawn_mismatch > 0) n_miss / total_drawn_mismatch else 1.0
attr(out, "retention_rates") <- list(r_match = r_match, r_mismatch = r_mismatch, ...)
```

**Between-site retention (lines 583–587)** — same pattern, plus tier-specific rates at lines 684–689.

**All match/mismatch partitions** (~15 locations: lines 269–270, 556–557, 641–642, 932–933, 941, 1001–1008, 1131–1136, 1644–1645):
```r
# CURRENT: matches <- batch[y == 0]
# PROPOSED: matches <- batch[y == 1]
```

### 2.2 `src/clesso_nn/export_for_nn.R` — lines 480–488

```r
# CURRENT:
rr[[paste0(nm, "_r0")]] <- sub$r_0
rr[[paste0(nm, "_r1")]] <- sub$r_1

# PROPOSED (Option B):
rr[[paste0(nm, "_r_match")]]    <- sub$r_match
rr[[paste0(nm, "_r_mismatch")]] <- sub$r_mismatch
```

This changes the **metadata.json** keys from e.g. `"within_r0"` to `"within_r_match"`.

### 2.3 `src/clesso_nn/model.py` — lines 981, 1247–1270

**Retention rate key mapping (lines 1247–1260)**:
```python
# CURRENT:
_rr_keys = {
    0: ("within_r0", "within_r1"),
    1: ("between_tier1_r0", "between_tier1_r1"),
    ...
}
r0_s = rr.get(r0_key, 1.0)
r1_s = rr.get(r1_key, 1.0)

# PROPOSED (Option B):
_rr_keys = {
    0: ("within_r_match", "within_r_mismatch"),
    1: ("between_tier1_r_match", "between_tier1_r_mismatch"),
    ...
}
r_match_s    = rr.get(rm_key, 1.0)
r_mismatch_s = rr.get(rmm_key, 1.0)
```

**Retrospective correction formula (line ~1267)**:
```python
# CURRENT (p = P(match), r0 = match retention, r1 = mismatch retention):
p_samp = r0_s * p_s / (r0_s * p_s + r1_s * (1.0 - p_s) + eps)

# PROPOSED — formula is UNCHANGED because p still represents P(match):
p_samp = r_match_s * p_s / (r_match_s * p_s + r_mismatch_s * (1.0 - p_s) + eps)
```

*The formula itself does not change* — `p` is the model's match probability regardless of how y is coded. What changes is only the *variable names* feeding into it.

---

## Phase 3: Loss formula — BCE coefficient swap

### 3.1 `src/clesso_nn/model.py` — line 1237

```python
# CURRENT (y=0 match, y=1 mismatch):
# When y=0: loss = -log(p)         → penalises low p for matches ✓
# When y=1: loss = -log(1-p)       → penalises high p for mismatches ✓
bce_per_pair = -(1.0 - y) * log_p - y * log_1m_p

# PROPOSED (y=1 match, y=0 mismatch):
# When y=1: loss = -log(p)         → penalises low p for matches ✓
# When y=0: loss = -log(1-p)       → penalises high p for mismatches ✓
bce_per_pair = -y * log_p - (1.0 - y) * log_1m_p
```

**Change**: swap the positions of `y` and `(1.0 - y)` in the BCE.

### 3.2 `src/clesso_nn/dataset.py` — line 628 (hard-pair hardness formula)

```python
# CURRENT:
h = -(1.0 - y) * torch.log(p) - y * torch.log(1.0 - p)

# PROPOSED:
h = -y * torch.log(p) - (1.0 - y) * torch.log(1.0 - p)
```

---

## Phase 4: Hard-pair mining — cell membership

### 4.1 `src/clesso_nn/dataset.py` — lines 551–555

```python
# CURRENT:
self.cell_match = self.y < 0.5    # y=0 → match
self.cell_mismatch = ~self.cell_match

# PROPOSED:
self.cell_match = self.y > 0.5    # y=1 → match
self.cell_mismatch = ~self.cell_match
```

The hard-pair distribution code (lines 703–745) uses `self.cell_match` / `self.cell_mismatch` and does **not** need further changes — it just needs the boolean masks to be correct.

---

## Phase 5: Match-boost pairs — stratum 4

### 5.1 `src/clesso_nn/model.py` — line 1279 (comment only)

```python
# CURRENT:
# Unweighted BCE on boosted match pairs (they are all y=0)

# PROPOSED:
# Unweighted BCE on boosted match pairs (they are all y=1)
```

The actual *logic* (`bce_per_pair[mask_boost]`) doesn't filter by y value — it filters by stratum. So no code change needed, just the comment.

---

## Phase 6: AUC and classification metrics

### 6.1 `src/clesso_nn/diagnostics.py` — lines 902–927

```python
# CURRENT:
# AUC: y=1 is mismatch, score = 1 - p_match (higher = more likely mismatch)
auc = _auc_wilcoxon(y_obs, 1 - p_match)
pred_class = (p_match < 0.5).astype(int)  # predict mismatch when p_match < 0.5

# PROPOSED:
# AUC: y=1 is match, score = p_match (higher = more likely match)
auc = _auc_wilcoxon(y_obs, p_match)
pred_class = (p_match >= 0.5).astype(int)  # predict match when p_match >= 0.5
```

**Confusion matrix labels** (lines 920–927):
```python
# CURRENT:
f"                Match(0)    Mismatch(1)",
f"  Pred Match      {tn:>8,}      {fn:>8,}",
f"  Pred Mismatch   {fp:>8,}      {tp:>8,}",

# PROPOSED:
f"                Mismatch(0)   Match(1)",
f"  Pred Mismatch   {tn:>8,}      {fn:>8,}",
f"  Pred Match      {fp:>8,}      {tp:>8,}",
```

### 6.2 `src/clesso_v2/clesso_diagnostics.R` — lines 269–280

```r
# CURRENT:
auc_val <- tryCatch(compute_auc(1 - p_hat, y_obs), ...)
y_pred_class <- as.integer(p_hat < thresh)

# PROPOSED:
auc_val <- tryCatch(compute_auc(p_hat, y_obs), ...)
y_pred_class <- as.integer(p_hat >= thresh)
```

---

## Phase 7: Diagnostic scripts (cosmetic label changes)

### 7.1 `scripts/diag_auc.py` — lines 18–19, 34–35

```python
# CURRENT:
print(f"  y=0 (match):    {(bp.y==0).sum():,}")
print(f"  y=1 (mismatch): {(bp.y==1).sum():,}")

# PROPOSED:
print(f"  y=0 (mismatch): {(bp.y==0).sum():,}")
print(f"  y=1 (match):    {(bp.y==1).sum():,}")
```

### 7.2 `scripts/diag_strata.py` — lines 15–21

Update display labels only. No logic changes.

### 7.3 `scripts/check_alpha_inflation.py` — line 20

```python
# CURRENT:
f"Overall match rate (y=0): {(within['y'] == 0).mean():.4f}"

# PROPOSED:
f"Overall match rate (y=1): {(within['y'] == 1).mean():.4f}"
```

---

## Phase 8: CLESSO v2 R pipeline (legacy)

All locations follow the same pattern: swap `y == 0` ↔ `y == 1` in filtering/counting.

| File | Lines | Change |
|------|-------|--------|
| `clesso_sampler.R` | 155–156, 189–190, 307–308, 410–411, 533–534 | Swap `y == 0` ↔ `y == 1` |
| `clesso_sampler_optimised.R` | 269–270, 556–557, 641–642, 932–933, 941, 1001–1008, 1131–1136, 1644–1645 | Swap `y == 0` ↔ `y == 1` |
| `run_clesso_alpha.R` | 119–123, 314 | Swap `y == 0` ↔ `y == 1` |
| `run_clesso_beta_fixAlpha.R` | 165–169, 316 | Swap `y == 0` ↔ `y == 1` |
| `clesso_diagnostics.R` | 190, 269–280 | Swap convention + AUC fix |
| `clesso_diagnostics_standalone.R` | 397, 430, 441 | Label swaps |
| `clesso_prepare_data.R` | 7 | Comment update |
| `clesso_config.R` | 113–121 | Comment updates only |

---

## Phase 9: Documentation (main.tex)

| Line(s) | Current text | Change |
|---------|-------------|--------|
| ~318 | `$y_p = 0$ if species identities match, $y_p = 1$ otherwise` | Swap to `$y_p = 1$ if match, $y_p = 0$ otherwise` |
| ~384–388 | Gradient asymmetry discussion (references y coefficient roles) | Update coefficient description |
| ~399–423 | Equations 4–5 (retrospective correction with r₀, r₁) | Rename to r_match, r_mismatch if using Option B |
| ~437–445 | Match-boost section: "they are all y=0" | Change to "they are all y=1" |
| ~1237 | BCE equation `$-(1-y)\log p - y\log(1-p)$` | Swap to `$-y\log p - (1-y)\log(1-p)$` |
| Various | Dataset docstring refs mentioning "0 (match) or 1 (mismatch)" | Swap wording |

---

## Phase 10: Tests

| File | Lines | Change |
|------|-------|--------|
| `tests/test_hard_mining.py` | 52 | Semantic update if test checks specific class behaviour |
| `tests/test_effort.py` | 78 | `torch.randint(0, 2, ...)` — unchanged, but verify test assertions |
| `tests/test_gdm_fixes.R` | 63, 85 | `wts <- ifelse(y == 0, 3, 1)` → `ifelse(y == 1, 3, 1)` |

---

## Phase 11: Existing data files

**Critical**: Any already-exported `.feather` pair files and `metadata.json` files contain the **old** y-coding and r0/r1 keys. These must be **re-exported** from the R pipeline after the swap, or a migration script must be written to flip y values in-place.

Options:
- **Re-export** (recommended): Re-run the full R pipeline (`obs_pair_sampler.R` → `clesso_sampler_optimised.R` → `export_for_nn.R`)
- **Migration script**: `pairs["y"] = 1 - pairs["y"]` on existing feather files, plus rename metadata.json keys

---

## Verification checklist

After implementing all changes:

- [ ] Re-export pairs from R pipeline (or run migration script)
- [ ] Verify `pairs.feather`: match pairs now have y=1, mismatch pairs have y=0
- [ ] Verify `metadata.json`: retention rate keys match new naming (r_match / r_mismatch)
- [ ] Train a model on a small subset and verify:
  - [ ] Loss converges (not diverging or constant)
  - [ ] AUC is comparable to pre-swap baseline (should be identical if done correctly)
  - [ ] Alpha predictions are in the same range as before
  - [ ] Eta values are non-degenerate (no collapse to 0 or 10)
- [ ] Run diagnostics and verify:
  - [ ] AUC computation no longer needs `1 - p_match` inversion
  - [ ] Confusion matrix labels are correct
  - [ ] Match-boost pairs (stratum 4) are all y=1
- [ ] Grep for any remaining `y == 0.*match` or `y == 1.*mismatch` patterns in the codebase
- [ ] Grep for any remaining `r_0`, `r_1`, `r0`, `r1` variable names (if using Option B naming)

## Risk assessment

**This is a high-risk refactor.** The y convention is deeply embedded across ~18 files and ~95 locations spanning two languages (R + Python). A single missed location would introduce a **silent bug** — the model would train with inverted gradients in one component, producing plausible but wrong results (e.g. eta collapse, alpha inflation, or degraded AUC).

**Recommended approach**: implement behind a feature flag first. Add a `y_convention = "match_is_one"` config option that applies `y = 1 - y` at load time in `dataset.py` (line 389) and swaps r0/r1 keys in `model.py`. This lets us test equivalence without touching 95 locations simultaneously.
