# Exact tree-leaf evaluation of the existing finite-background grouped game.
# This preserves both exact Shapley and the original sampled-order estimator.
load_leaf_backend <- function(cache) {
  dir.create(cache, recursive = TRUE, showWarnings = FALSE)
  source <- "scripts/nonlinear_ld_leaf_shap.c"
  hash <- unname(tools::md5sum(source))
  library <- file.path(cache, paste0("leaf_", hash, .Platform$dynlib.ext))
  if (!file.exists(library)) {
    build <- file.path(cache, paste0("build_", hash))
    dir.create(build, showWarnings = FALSE)
    stopifnot(file.copy(source, file.path(build, "leaf.c"), overwrite = TRUE))
    previous <- setwd(build)
    on.exit(setwd(previous))
    status <- system2(file.path(R.home("bin"), "R"), c("CMD", "SHLIB", "leaf.c"))
    if (status != 0L) stop("Could not build grouped Shapley backend")
    setwd(previous)
    stopifnot(file.copy(file.path(build, paste0("leaf", .Platform$dynlib.ext)), library))
  }
  dyn.load(normalizePath(library))
  invisible(library)
}

group_shap_leaf <- function(model, background, evaluation, paths = 16L, seed = 1L, exact = FALSE) {
  stopifnot(model$treetype == "Regression", ncol(background) == BLOCKS * SNPS,
            ncol(evaluation) == BLOCKS * SNPS, all(is.finite(background)), all(is.finite(evaluation)))
  set.seed(seed)
  orders <- matrix(0L, 2L * paths, BLOCKS)
  needed <- c(0L, 2L^BLOCKS - 1L)
  for (i in seq_len(paths)) {
    order <- sample.int(BLOCKS)
    orders[2L * i - 1L, ] <- order
    orders[2L * i, ] <- rev(order)
    for (sequence in list(order, rev(order)))
      needed <- c(needed, cumsum(bitwShiftL(1L, sequence - 1L)))
  }
  storage.mode(background) <- storage.mode(evaluation) <- "double"
  phi <- .Call("nonlinear_ld_leaf_shap", model$forest, background, evaluation, orders,
               as.logical(exact), rep.int(seq_len(BLOCKS) - 1L, rep.int(SNPS, BLOCKS)))
  full <- predict(model, data = evaluation, num.threads = 1L)$predictions
  empty <- mean(predict(model, data = background, num.threads = 1L)$predictions)
  error <- max(abs(rowSums(phi) - (full - empty)))
  if (!is.finite(error) || error > 1e-8) stop("Leaf grouped Shapley efficiency check failed: ", error)
  list(scores = colMeans(abs(phi)), phi = phi, efficiency_error = error,
       coalition_count = if (exact) as.integer(2L^BLOCKS) else length(unique(needed)), exact = exact)
}
