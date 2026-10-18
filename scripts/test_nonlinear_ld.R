#!/usr/bin/env Rscript
source("scripts/run_nonlinear_ld.R")
assert_close <- function(x, y, tolerance = 1e-10) stopifnot(max(abs(x - y)) < tolerance)

# Population moments of additive and dominance encodings under HWE.
for (p in c(.1, .25, .5)) {
  e <- encodings(0:2, p)
  probability <- c((1 - p)^2, 2 * p * (1 - p), p^2)
  assert_close(sum(probability * e$A), 0)
  assert_close(sum(probability * e$D), 0)
  assert_close(sum(probability * e$A^2), 1)
  assert_close(sum(probability * e$D^2), 1)
  assert_close(sum(probability * e$A * e$D), 0)
}
cat("PASS: population encoding moments\n")

for (scenario in c(4L, 5L)) {
  for (strength in c(0, 1)) {
    low <- generate_data(scenario, strength, 0, 71L, n = 300L)
    high <- generate_data(scenario, strength, .99, 71L, n = 300L)
    for (key in c("y", "genetic", "beta", "train", "test")) stopifnot(identical(low[[key]], high[[key]]))
    first <- low$x[, seq.int(1L, BLOCKS * SNPS, by = SNPS)]
    a <- (first - 1) / sqrt(.5)
    d <- 2 * (first == 1) - 1
    expected <- if (scenario == 4L) drop((sqrt(1 - strength) * a[, 1:5] + sqrt(strength) * d[, 1:5]) %*% low$beta) else
      .5 * a[, 1L] + strength * a[, 1L] * a[, 2L]
    assert_close(low$genetic, expected)
    stopifnot(!length(intersect(low$train, low$test)), all(low$x %in% 0:2))
  }
}
cat("PASS: phenotype definitions, matched sweeps and partition separation\n")

# Check the actual mvtnorm + mixing generator's genotype frequencies and LD.
large <- generate_data(4L, 1, .9, 32L, n = 40000L)
stopifnot(max(abs(tabulate(large$x[, 1L] + 1L, 3L) / nrow(large$x) - c(.25, .5, .25))) < .015)
# Independent numerical bivariate Gaussian calculation of dosage covariance.
threshold <- qnorm(c(.25, .75))
joint <- sum(vapply(threshold, function(a) sum(vapply(threshold, function(b)
  mvtnorm::pmvnorm(lower = c(a, b), upper = c(Inf, Inf),
                   sigma = matrix(c(1, .9, .9, 1), 2L))[1L], numeric(1L))), numeric(1L)))
expected_cor <- (joint - 1) / .5
stopifnot(abs(cor(large$x[, 1L], large$x[, 5L * SNPS + 1L]) - expected_cor) < .02,
          abs(cor(large$x[, 1L], large$x[, SNPS + 1L])) < .025)
rm(large)
cat("PASS: mvtnorm HWE marginals, proxy LD and source independence\n")

set.seed(51L)
x <- matrix(rbinom(500L * BLOCKS * SNPS, 2L, .5), 500L)
y <- .4 * x[, 1L] + rnorm(500L)
scores <- linear_gwas(x, y)
direct <- vapply(seq_len(BLOCKS * SNPS), function(j)
  summary(lm(y ~ x[, j]))$coefficients[2L, 4L], numeric(1L))
expected <- -log10(vapply(seq_len(BLOCKS), function(b)
  min(direct[seq.int((b - 1L) * SNPS + 1L, b * SNPS)]), numeric(1L)))
assert_close(scores, expected)
assert_close(recovery(rep(1, 10L), 1:5), .5)
assert_close(recovery(c(4, 4, rep(1, 8L)), 1L), .5)
cat("PASS: GWAS vs independent lm fits; boundary tie credit\n")

# Analytic Shapley values for a two-group interaction, with unused exact proxies.
set.seed(5L)
bg <- matrix(rnorm(8L * BLOCKS * SNPS), 8L)
ev <- matrix(rnorm(5L * BLOCKS * SNPS), 5L)
bg[, (2L * SNPS + 1L):(3L * SNPS)] <- bg[, 1:SNPS]
ev[, (2L * SNPS + 1L):(3L * SNPS)] <- ev[, 1:SNPS]
oracle <- function(x) .5 * x[, 1L] + 2 * x[, 1L] * x[, SNPS + 1L]
v0 <- .5 * ev[, 1L] + 2 * ev[, 1L] * mean(bg[, SNPS + 1L])
v1 <- .5 * mean(bg[, 1L]) + 2 * mean(bg[, 1L]) * ev[, SNPS + 1L]
empty <- mean(oracle(bg))
full <- oracle(ev)
expected <- cbind(.5 * (v0 - empty + full - v1), .5 * (v1 - empty + full - v0))
for (exact in c(FALSE, TRUE)) {
  shap <- group_shap(oracle, bg, ev, paths = 16L, seed = 11L, exact = exact)
  assert_close(shap$phi[, 1:2], expected)
  assert_close(shap$phi[, 3:10], 0)
  stopifnot(shap$efficiency_error < 1e-10)
}
cat("PASS: analytic joint-row interventional Shapley, dummy groups and efficiency\n")

colnames(x) <- sprintf("snp_%03d", seq_len(ncol(x)))
fit <- fit_rf(x, y, 10L, 61L)
stopifnot(fit$mtry == ncol(x), fit$num.trees == 10L,
          all(is.finite(predict(fit, data = x, num.threads = 1L)$predictions)))
# Reproducible fits with fixed tree seed.
fit2 <- fit_rf(x, y, 10L, 61L)
assert_close(predict(fit, data = x, num.threads = 1L)$predictions,
             predict(fit2, data = x, num.threads = 1L)$predictions)
cat("PASS: ranger settings and deterministic fits\nAll six scientific checks passed.\n")
