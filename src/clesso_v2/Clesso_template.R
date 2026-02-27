library(TMB)
library(data.table)

# ---- Inputs assumed ----
# site_pairs: site_i, site_j, within_between, and y or match
# covariates: pairwise covariates (same row order as site_pairs) -> X
# site_cov: site-level covariates for alpha (one row per site) -> Z

sp <- as.data.table(site_pairs)
Xdf <- as.data.table(covariates)
sc <- as.data.table(site_cov)

stopifnot(nrow(sp) == nrow(Xdf))

# Response y: 0=match, 1=mismatch
if (!("y" %in% names(sp))) {
  stopifnot("match" %in% names(sp))
  sp[, y := as.integer(!match)]
} else {
  sp[, y := as.integer(y)]
}
stopifnot(all(sp$y %in% c(0L, 1L)))

# Within flag
sp[, is_within := as.integer(within_between)]
stopifnot(all(sp$is_within %in% c(0L, 1L)))

# ---- Pair weights ----
# If you already have weights, use them. Otherwise default to 1.
if (!("w" %in% names(sp))) sp[, w := 1.0]
sp[, w := as.numeric(w)]

# ---- Map site IDs to contiguous 0-based indices ----
all_sites <- unique(c(sp$site_i, sp$site_j))
site_map <- data.table(site_id = all_sites)[, site_index := .I - 1L]

sp <- merge(sp, site_map, by.x="site_i", by.y="site_id", all.x=TRUE, sort=FALSE)
setnames(sp, "site_index", "site_i_idx")

sp <- merge(sp, site_map, by.x="site_j", by.y="site_id", all.x=TRUE, sort=FALSE)
setnames(sp, "site_index", "site_j_idx")

nSites <- nrow(site_map)

# ---- Build X matrix for turnover ----
X <- as.matrix(Xdf)
storage.mode(X) <- "double"

# ---- Build Z matrix for alpha regression ----
# site_cov must include all sites in site_map (or at least those that appear in pairs)
# Merge site_map with site_cov to ensure correct ordering.
stopifnot("site_id" %in% names(sc))
scm <- merge(site_map, sc, by="site_id", all.x=TRUE, sort=FALSE)

# Choose which columns are alpha covariates (exclude site_id and site_index)
alpha_cov_cols <- setdiff(names(scm), c("site_id", "site_index"))
stopifnot(length(alpha_cov_cols) > 0)

Z <- as.matrix(scm[, ..alpha_cov_cols])
storage.mode(Z) <- "double"

# Optional: standardize Z for stability
Z <- scale(Z)

# ---- Data list for TMB ----
data_list <- list(
  y         = as.numeric(sp$y),
  site_i    = as.integer(sp$site_i_idx),
  site_j    = as.integer(sp$site_j_idx),
  is_within = as.integer(sp$is_within),
  X         = X,
  w         = as.numeric(sp$w),
  Z         = Z
)

Kbeta  <- ncol(X)
Kalpha <- ncol(Z)

# ---- Parameters (initial values) ----
parameters <- list(
  # turnover
  eta0_raw   = 0,
  beta_raw   = rep(log(0.01), Kbeta),

  # alpha regression
  alpha0      = log(20 - 1),            # so alpha ~ 20 at mean(Z)=0 when u=0
  theta_alpha = rep(0, Kalpha),
  u_site      = rep(0, nSites),
  log_sigma_u = log(0.5)
)

# ---- Compile / load ----
cpp_file <- "pair_alpha_beta.cpp"
compile(cpp_file)
dyn.load(dynlib("pair_alpha_beta"))

obj <- MakeADFun(
  data = data_list,
  parameters = parameters,
  DLL = "pair_alpha_beta",
  silent = TRUE
)

fit <- nlminb(
  start = obj$par,
  objective = obj$fn,
  gradient  = obj$gr,
  control = list(eval.max = 4000, iter.max = 4000)
)

rep <- sdreport(obj)

# Extract reports
est <- summary(rep, "report")
alpha_rows <- grep("^alpha_site", rownames(est), value = TRUE)
logalpha_rows <- grep("^log_alpha_site", rownames(est), value = TRUE)
beta_rows <- grep("^beta", rownames(est), value = TRUE)

results <- list(
  fit = fit,
  sdreport = rep,
  alpha_site = est[alpha_rows, , drop=FALSE],
  log_alpha_site = est[logalpha_rows, , drop=FALSE],
  beta = est[beta_rows, , drop=FALSE],
  site_map = site_map,
  alpha_cov_cols = alpha_cov_cols
)

results