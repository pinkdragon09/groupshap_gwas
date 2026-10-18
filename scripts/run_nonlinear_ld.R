#!/usr/bin/env Rscript
# Independent R reproduction of the nonlinear LD benchmark. Run from repo root:
# Rscript scripts/test_nonlinear_ld.R
# Rscript scripts/run_nonlinear_ld.R --reps 100 --jobs 12
# Rscript scripts/report_nonlinear_ld.R outputs/nonlinear_ld/20261017_r_mvtnorm
# Install mvtnorm and ranger in the normal library or .Rlib-nonlinear first.

if (dir.exists(".Rlib-nonlinear")) .libPaths(c(normalizePath(".Rlib-nonlinear"), .libPaths()))
for (pkg in c("mvtnorm", "ranger", "jsonlite")) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop("Install R package: ", pkg)
}
BLOCKS <- 10L
SNPS <- 20L
METHODS <- c("Linear GWAS", "RF + interventional groupSHAP")
ROOT <- normalizePath(".")
RUNNER <- file.path(ROOT, "scripts/run_nonlinear_ld.R")

encodings <- function(x, p) {
  list(A = (x - 2 * p) / sqrt(2 * p * (1 - p)),
       D = ((x == 1) - 2 * p * (1 - p) - (1 - 2 * p) * (x - 2 * p)) /
         (2 * p * (1 - p)))
}

generate_data <- function(scenario, strength, rho, seed, n = 1000L,
                          within = .6, h2 = .5, maf_mode = "balanced") {
  stopifnot(scenario %in% c(4L, 5L), rho >= 0, rho < 1, within >= 0, within < 1,
            n >= 50, h2 > 0, h2 < 1, strength >= 0,
            scenario != 4L || strength <= 1, maf_mode %in% c("balanced", "variable"))
  set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  sigma <- within ^ abs(outer(seq_len(SNPS), seq_len(SNPS), "-"))
  # Independent AR(1) Gaussian vectors, then a linear Gaussian transformation.
  # Cov(Z_source[j], Z_proxy[k]) = rho * sigma[j,k].
  independent <- lapply(seq_len(BLOCKS), function(b)
    mvtnorm::rmvnorm(n, sigma = sigma, method = "chol"))
  latent <- independent
  mafs <- matrix(runif(BLOCKS * SNPS, .1, .45), BLOCKS, SNPS)
  if (maf_mode == "balanced") mafs[,] <- .5
  proxies <- if (scenario == 4L) setNames(1:5, 6:10) else setNames(1:2, 3:4)
  for (proxy_name in names(proxies)) {
    proxy <- as.integer(proxy_name)
    source <- proxies[[proxy_name]]
    latent[[proxy]] <- rho * independent[[source]] + sqrt(1 - rho^2) * independent[[proxy]]
    mafs[proxy, ] <- mafs[source, ]
  }
  dosage <- lapply(seq_len(BLOCKS), function(b) {
    low <- qnorm((1 - mafs[b, ])^2)
    high <- qnorm(1 - mafs[b, ]^2)
    1 * (sweep(latent[[b]], 2L, low, ">") + sweep(latent[[b]], 2L, high, ">"))
  })
  x <- do.call(cbind, dosage)
  colnames(x) <- sprintf("snp_%03d", seq_len(ncol(x)))
  encoded <- lapply(seq_len(BLOCKS), function(b) encodings(dosage[[b]][, 1L], mafs[b, 1L]))
  a <- do.call(cbind, lapply(encoded, `[[`, "A"))
  d <- do.call(cbind, lapply(encoded, `[[`, "D"))
  signs <- sample(c(-1, 1), 5L, replace = TRUE)
  if (scenario == 4L) {
    beta <- signs / sqrt(5)
    genetic <- drop((sqrt(1 - strength) * a[, 1:5] + sqrt(strength) * d[, 1:5]) %*% beta)
    causal <- 1:5
  } else {
    beta <- c(.5, strength)
    genetic <- .5 * a[, 1L] + strength * a[, 1L] * a[, 2L]
    causal <- if (strength > 0) 1:2 else 1L
  }
  order <- sample.int(n)
  ntrain <- as.integer(.8 * n)
  train <- order[seq_len(ntrain)]
  test <- order[seq.int(ntrain + 1L, n)]
  # Population variance divisor, matching numpy.var(ddof=0).
  noise_sd <- sqrt(mean((genetic[train] - mean(genetic[train]))^2) * (1 - h2) / h2)
  y <- genetic + noise_sd * rnorm(n)
  list(x = x, y = y, genetic = genetic, beta = beta, causal = causal, proxies = proxies,
       train = train, test = test, mafs = mafs, noise_sd = noise_sd)
}

linear_gwas <- function(x, y) {
  # Intercept + dosage, two-sided t test of the dosage coefficient.
  xc <- sweep(x, 2L, colMeans(x), "-")
  yc <- y - mean(y)
  denominator <- sqrt(colSums(xc^2) * sum(yc^2))
  r <- drop(crossprod(yc, xc)) / denominator
  r[!is.finite(r)] <- 0
  r <- pmin(1, pmax(-1, r))
  tstat <- abs(r) * sqrt((length(y) - 2) / pmax(1 - r^2, .Machine$double.xmin))
  pvalues <- 2 * pt(tstat, df = length(y) - 2, lower.tail = FALSE)
  vapply(seq_len(BLOCKS), function(b) {
    cols <- seq.int((b - 1L) * SNPS + 1L, b * SNPS)
    -log10(max(min(pvalues[cols]), 1e-300))
  }, numeric(1L))
}

recovery <- function(scores, causal) {
  k <- length(causal)
  cutoff <- sort(scores, decreasing = TRUE)[k]
  tied <- abs(scores - cutoff) <= 1e-12 + 1e-10 * abs(cutoff)
  above <- scores > cutoff & !tied
  credit <- as.numeric(above) + tied * (k - sum(above)) / sum(tied)
  sum(credit[causal]) / k
}

r2 <- function(y, prediction) 1 - sum((y - prediction)^2) / sum((y - mean(y))^2)

group_shap <- function(predict_fun, background, evaluation, paths = 16L,
                       seed = 1L, exact = FALSE) {
  stopifnot(ncol(background) == BLOCKS * SNPS, ncol(evaluation) == BLOCKS * SNPS)
  set.seed(seed)
  orders <- vector("list", 2L * paths)
  needed <- c(0L, 2L^BLOCKS - 1L)
  for (i in seq_len(paths)) {
    order <- sample.int(BLOCKS)
    orders[[2L * i - 1L]] <- order
    orders[[2L * i]] <- rev(order)
    for (sequence in list(order, rev(order))) {
      needed <- c(needed, cumsum(bitwShiftL(1L, sequence - 1L)))
    }
  }
  masks <- if (exact) 0:(2L^BLOCKS - 1L) else sort(unique(needed))
  ne <- nrow(evaluation)
  nb <- nrow(background)
  full <- 2L^BLOCKS - 1L
  values <- matrix(NA_real_, 2L^BLOCKS, ne)
  values[1L, ] <- mean(predict_fun(background))
  values[full + 1L, ] <- predict_fun(evaluation)
  bg_rows <- rep(seq_len(nb), times = ne)
  ev_rows <- rep(seq_len(ne), each = nb)
  pending <- setdiff(masks, c(0L, full))
  for (start in seq.int(1L, length(pending), by = 16L)) {
    batch <- pending[seq.int(start, min(start + 15L, length(pending)))]
    synthetic <- background[rep(bg_rows, times = length(batch)), , drop = FALSE]
    for (j in seq_along(batch)) {
      rows <- seq.int((j - 1L) * ne * nb + 1L, j * ne * nb)
      for (group in seq_len(BLOCKS)) {
        if (bitwAnd(batch[j], bitwShiftL(1L, group - 1L)) != 0L) {
          cols <- seq.int((group - 1L) * SNPS + 1L, group * SNPS)
          synthetic[rows, cols] <- evaluation[ev_rows, cols, drop = FALSE]
        }
      }
    }
    prediction <- predict_fun(synthetic)
    block_means <- colMeans(matrix(prediction, nb, ne * length(batch)))
    values[batch + 1L, ] <- t(matrix(block_means, ne, length(batch)))
  }
  phi <- matrix(0, ne, BLOCKS)
  if (exact) {
    sizes <- vapply(0:full, function(mask) sum(as.integer(intToBits(mask))[seq_len(BLOCKS)]), integer(1L))
    for (group in seq_len(BLOCKS)) {
      bit <- bitwShiftL(1L, group - 1L)
      absent <- (0:full)[bitwAnd(0:full, bit) == 0L]
      weight <- 1 / (BLOCKS * choose(BLOCKS - 1L, sizes[absent + 1L]))
      phi[, group] <- colSums((values[absent + bit + 1L, , drop = FALSE] -
                               values[absent + 1L, , drop = FALSE]) * weight)
    }
  } else {
    for (order in orders) {
      mask <- 0L
      for (group in order) {
        added <- mask + bitwShiftL(1L, group - 1L)
        phi[, group] <- phi[, group] + (values[added + 1L, ] - values[mask + 1L, ]) / length(orders)
        mask <- added
      }
    }
  }
  error <- max(abs(rowSums(phi) - (values[full + 1L, ] - values[1L, ])))
  if (!is.finite(error) || error > 1e-8) stop("Shapley efficiency check failed: ", error)
  list(scores = colMeans(abs(phi)), phi = phi, efficiency_error = error,
       coalition_count = length(masks), exact = exact)
}

fit_rf <- function(x, y, trees, seed) {
  ranger::ranger(x = x, y = y, num.trees = trees, mtry = ncol(x),
                 min.node.size = 1L, min.bucket = 5L, splitrule = "variance",
                 replace = TRUE, sample.fraction = 1, importance = "none",
                 oob.error = FALSE, num.threads = 1L, seed = seed)
}

setting_key <- function(scenario, strength, rho, rep) {
  sprintf("s%d_strength%g_rho%g_rep%03d", scenario, strength, rho, rep)
}

run_one <- function(task, config, out) {
  scenario <- task$scenario
  strength <- task$strength
  rho <- task$rho
  rep <- task$replicate
  target <- file.path(out, "replicates", paste0(setting_key(scenario, strength, rho, rep), ".rds"))
  if (file.exists(target)) return(readRDS(target))
  started <- proc.time()[[3L]]
  seed <- config$seed_base + rep
  data <- generate_data(scenario, strength, rho, seed, config$n, config$within, config$h2, config$maf_mode)
  train <- data$train
  test <- data$test
  model <- fit_rf(data$x[train, , drop = FALSE], data$y[train], config$trees, seed)
  predict_fun <- function(newdata) predict(model, data = newdata, num.threads = 1L)$predictions
  set.seed(seed + 50000L)
  background <- sample(train, config$background)
  evaluation <- sample(test, config$evaluation)
  shap <- group_shap(predict_fun, data$x[background, , drop = FALSE], data$x[evaluation, , drop = FALSE],
                     paths = config$paths, seed = seed + 60000L, exact = scenario == 5L && strength > 0)
  scores <- setNames(list(linear_gwas(data$x[train, , drop = FALSE], data$y[train]), shap$scores), METHODS)
  observed <- vapply(names(data$proxies), function(p) {
    s <- data$proxies[[p]]
    cor(data$x[test, (s - 1L) * SNPS + 1L], data$x[test, (as.integer(p) - 1L) * SNPS + 1L])
  }, numeric(1L))
  common <- list(scenario = scenario, strength = strength, rho_between = rho, replicate = rep,
                 maf_mode = config$maf_mode, n = config$n,
                 rf_test_r2 = r2(data$y[test], predict_fun(data$x[test, , drop = FALSE])),
                 oracle_test_r2 = r2(data$y[test], data$genetic[test]),
                 observed_paired_snp_correlation = mean(observed),
                 shapley_efficiency_error = shap$efficiency_error)
  rows <- do.call(rbind, lapply(names(scores), function(method) {
    s <- scores[[method]]
    total <- sum(abs(s))
    causal <- data$causal
    active_proxies <- as.integer(names(data$proxies)[data$proxies %in% causal])
    independent_null <- setdiff(seq_len(BLOCKS), c(causal, active_proxies))
    as.data.frame(c(common, list(method = method, recovery_at_k = recovery(s, causal),
      causal_score_fraction = sum(abs(s[causal])) / total,
      correlated_null_score_fraction = sum(abs(s[active_proxies])) / total,
      independent_null_score_fraction = sum(abs(s[independent_null])) / total)))
  }))
  record <- list(rows = rows, scores = scores, phi = shap$phi,
                 coalition_count = shap$coalition_count, exact_shapley = shap$exact,
                 beta = data$beta, causal = data$causal, proxies = data$proxies,
                 maf_first = data$mafs[, 1L], noise_sd = data$noise_sd,
                 train_indices = train, test_indices = test,
                 background_indices = background, evaluation_indices = evaluation,
                 elapsed_seconds = proc.time()[[3L]] - started)
  temporary <- paste0(target, ".tmp")
  saveRDS(record, temporary)
  if (!file.rename(temporary, target)) stop("Could not save replicate: ", target)
  record
}

summarize_records <- function(records, out) {
  frame <- do.call(rbind, lapply(records, `[[`, "rows"))
  frame <- frame[order(frame$scenario, frame$strength, frame$rho_between, frame$replicate, frame$method), ]
  write.csv(frame, file.path(out, "replicate_metrics.csv"), row.names = FALSE)
  keys <- c("scenario", "strength", "rho_between", "method", "maf_mode", "n")
  metrics <- c("recovery_at_k", "causal_score_fraction", "correlated_null_score_fraction",
               "independent_null_score_fraction", "rf_test_r2", "oracle_test_r2",
               "observed_paired_snp_correlation")
  groups <- split(seq_len(nrow(frame)), interaction(frame[keys], drop = TRUE))
  summary <- do.call(rbind, lapply(groups, function(idx) {
    part <- frame[idx, , drop = FALSE]
    row <- as.list(part[1L, keys])
    row$replicates <- nrow(part)
    for (metric in metrics) {
      row[[paste0(metric, "_mean")]] <- mean(part[[metric]])
      row[[paste0(metric, "_sd")]] <- sd(part[[metric]])
      row[[paste0(metric, "_se")]] <- sd(part[[metric]]) / sqrt(nrow(part))
    }
    as.data.frame(row)
  }))
  summary <- summary[order(summary$scenario, summary$strength, summary$rho_between, summary$method), ]
  rownames(summary) <- NULL
  write.csv(summary, file.path(out, "summary.csv"), row.names = FALSE)
  summary
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  options <- list(reps = 100L, jobs = 12L, n = 1000L, trees = 100L, within = .6, h2 = .5,
                  background = 16L, evaluation = 32L, paths = 16L,
                  rhos = c(0, .6, .9, .99), dominance_fractions = c(0, .5, 1),
                  interaction_strengths = c(0, .5, 1, 2), maf_mode = "balanced",
                  seed_base = 2026101800L, out = "outputs/nonlinear_ld/20261017_r_mvtnorm")
  if (length(args) %% 2L != 0L) stop("Options need --name value pairs; lists are comma-separated")
  if (length(args)) for (i in seq.int(1L, length(args), by = 2L)) {
    name <- gsub("-", "_", sub("^--", "", args[i]))
    if (!name %in% names(options)) stop("Unknown option: ", args[i])
    value <- args[i + 1L]
    options[[name]] <- if (name %in% c("out", "maf_mode")) value else as.numeric(strsplit(value, ",", fixed = TRUE)[[1L]])
  }
  stopifnot(options$reps >= 2, options$jobs >= 1, options$trees >= 1, options$paths >= 1,
            options$background > 0, options$background <= as.integer(.8 * options$n),
            options$evaluation > 0, options$evaluation <= options$n - as.integer(.8 * options$n),
            all(options$rhos >= 0 & options$rhos < 1))
  out <- options$out
  jobs <- options$jobs
  config <- options[setdiff(names(options), c("out", "jobs"))]
  config$methods <- METHODS
  config$rf_settings <- list(mtry = BLOCKS * SNPS, min_node_size = 1L, min_bucket = 5L,
                             replace = TRUE, sample_fraction = 1, splitrule = "variance")
  config$source_md5 <- unname(tools::md5sum(RUNNER))
  config$versions <- list(R = R.version.string, mvtnorm = as.character(packageVersion("mvtnorm")),
                         ranger = as.character(packageVersion("ranger")), jsonlite = as.character(packageVersion("jsonlite")))
  dir.create(file.path(out, "replicates"), recursive = TRUE, showWarnings = FALSE)
  manifest <- file.path(out, "config.rds")
  if (file.exists(manifest) && !identical(readRDS(manifest), config)) stop("Config/source changed; use a fresh output directory")
  saveRDS(config, manifest)
  jsonlite::write_json(config, file.path(out, "config.json"), auto_unbox = TRUE, pretty = TRUE, digits = 16)
  writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
  settings <- rbind(expand.grid(scenario = 4L, strength = config$dominance_fractions, rho = config$rhos,
                                replicate = seq_len(config$reps) - 1L),
                    expand.grid(scenario = 5L, strength = config$interaction_strengths, rho = config$rhos,
                                replicate = seq_len(config$reps) - 1L))
  tasks <- lapply(seq_len(nrow(settings)), function(i) as.list(settings[i, ]))
  cat(nrow(settings), "fits;", jobs, "workers; output", out, "\n")
  flush.console()
  started <- proc.time()[[3L]]
  for (start in seq.int(1L, length(tasks), by = jobs)) {
    batch <- tasks[seq.int(start, min(start + jobs - 1L, length(tasks)))]
    result <- parallel::mclapply(batch, run_one, config = config, out = out,
                                 mc.cores = jobs, mc.set.seed = FALSE, mc.preschedule = FALSE)
    failed <- vapply(result, inherits, logical(1L), "try-error")
    if (any(failed)) stop(paste(result[failed], collapse = "\n"))
    cat(sprintf("Completed %d/%d; elapsed %.1fs\n", min(start + jobs - 1L, length(tasks)),
                length(tasks), proc.time()[[3L]] - started))
    flush.console()
  }
  records <- lapply(tasks, run_one, config = config, out = out)
  summary <- summarize_records(records, out)
  print(summary[c("scenario", "strength", "rho_between", "method", "recovery_at_k_mean", "rf_test_r2_mean")], row.names = FALSE)
}

if (sys.nframe() == 0L) main()
