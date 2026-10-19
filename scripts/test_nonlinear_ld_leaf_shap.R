#!/usr/bin/env Rscript
source("scripts/run_nonlinear_ld.R")
source("scripts/nonlinear_ld_leaf_shap.R")
load_leaf_backend("/tmp/nonlinear_ld_leaf_backend")
maximum <- 0
timing <- list()
for (within in c(.6, .9, .99)) for (scenario in c(4L, 5L)) {
  data <- generate_data(scenario, 1, .3, 2026101800L, within = within)
  model <- fit_rf(data$x[data$train, ], data$y[data$train], 100L, 2026101800L)
  predict_fun <- function(x) predict(model, data = x, num.threads = 1L)$predictions
  set.seed(2026151800L)
  bg <- data$x[sample(data$train, 16L), ]
  ev <- data$x[sample(data$test, 32L), ]
  for (exact in c(FALSE, TRUE)) {
    elapsed_reference <- system.time(reference <- group_shap(predict_fun, bg, ev, seed = 62L, exact = exact))["elapsed"]
    elapsed_leaf <- system.time(leaf <- group_shap_leaf(model, bg, ev, seed = 62L, exact = exact))["elapsed"]
    error <- max(abs(reference$phi - leaf$phi))
    maximum <- max(maximum, error)
    stopifnot(error < 1e-10, max(abs(reference$scores - leaf$scores)) < 1e-10,
              identical(reference$coalition_count, leaf$coalition_count), leaf$efficiency_error < 1e-10,
              identical(recovery(reference$scores, data$causal), recovery(leaf$scores, data$causal)))
    timing[[length(timing) + 1L]] <- data.frame(within, scenario, exact,
                                              reference_seconds = unname(elapsed_reference),
                                              leaf_seconds = unname(elapsed_leaf), maximum_phi_error = error)
    cat(sprintf("PASS: within %g, scenario %d, exact=%s; max error %.3g; %.2fs vs %.2fs\n",
                within, scenario, exact, error, elapsed_reference, elapsed_leaf))
    flush.console()
  }
}
write.csv(do.call(rbind, timing), "/tmp/nonlinear_ld_leaf_equivalence.csv", row.names = FALSE)
cat(sprintf("All twelve reference/leaf comparisons passed; maximum error %.3g\n", maximum))
