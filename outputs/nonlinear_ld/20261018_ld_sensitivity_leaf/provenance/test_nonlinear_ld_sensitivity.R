#!/usr/bin/env Rscript
source("scripts/test_nonlinear_ld.R")

# With Cholesky generation, changing within-block LD keeps the causal first
# SNPs fixed. Test the intended matched design across BOTH LD dimensions.
for (scenario in c(4L, 5L)) {
  for (strength in if (scenario == 4L) c(0, .5, 1) else c(0, .5, 1, 2)) {
    reference <- generate_data(scenario, strength, 0, 71L, n = 300L, within = .6)
    for (within in c(.6, .9, .99)) for (rho in c(0, .3, .6, .9, .99)) {
      comparison <- generate_data(scenario, strength, rho, 71L, n = 300L, within = within)
      for (key in c("y", "genetic", "beta", "train", "test", "noise_sd"))
        stopifnot(identical(reference[[key]], comparison[[key]]))
    }
  }
}
cat("PASS: causal SNP effects, phenotype, noise and partitions matched across both LD dimensions\n")

# Independent bivariate-normal integrals check the dosage correlations after
# thresholding, including high within-block controls and the new rho=0.3.
expected_dosage_cor <- function(rho) {
  threshold <- qnorm(c(.25, .75))
  joint <- sum(vapply(threshold, function(a) sum(vapply(threshold, function(b)
    mvtnorm::pmvnorm(lower = c(a, b), upper = c(Inf, Inf),
                     sigma = matrix(c(1, rho, rho, 1), 2L))[1L], numeric(1L))), numeric(1L)))
  (joint - 1) / .5
}
for (within in c(.6, .9, .99)) {
  data <- generate_data(4L, 1, .3, 39L, n = 20000L, within = within)
  stopifnot(abs(cor(data$x[, 1L], data$x[, 2L]) - expected_dosage_cor(within)) < .025,
            abs(cor(data$x[, 1L], data$x[, SNPS]) - expected_dosage_cor(within^(SNPS - 1L))) < .025,
            abs(cor(data$x[, 1L], data$x[, 5L * SNPS + 1L]) - expected_dosage_cor(.3)) < .025,
            abs(cor(data$x[, 1L], data$x[, SNPS + 1L])) < .025)
}
cat("PASS: new within-block controls and rho=0.3 match theoretical dosage LD\n")
cat("All eight scientific checks passed.\n")
