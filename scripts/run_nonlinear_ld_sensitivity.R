#!/usr/bin/env Rscript
# Expand the frozen R benchmark without changing its generator or fitting code.
source("scripts/run_nonlinear_ld.R")

main_sensitivity <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opt <- list(reps = 100L, jobs = 12L, within = c(.6, .9, .99),
              rhos = c(0, .3, .6, .9, .99),
              out = "outputs/nonlinear_ld/20261018_ld_sensitivity",
              reuse = "outputs/nonlinear_ld/20261017_r_mvtnorm")
  if (length(args) %% 2L) stop("Use --name value pairs")
  if (length(args)) for (i in seq.int(1L, length(args), by = 2L)) {
    key <- sub("^--", "", args[i])
    if (!key %in% names(opt)) stop("Unknown option: ", key)
    opt[[key]] <- if (key %in% c("out", "reuse")) args[i + 1L] else
      as.numeric(strsplit(args[i + 1L], ",", fixed = TRUE)[[1L]])
  }
  stopifnot(length(opt$reps) == 1L, opt$reps >= 2, opt$reps == as.integer(opt$reps),
            length(opt$jobs) == 1L, opt$jobs >= 1, opt$jobs == as.integer(opt$jobs),
            length(opt$within) > 0, !anyDuplicated(opt$within),
            all(is.finite(opt$within) & opt$within >= 0 & opt$within < 1),
            length(opt$rhos) > 0, !anyDuplicated(opt$rhos),
            all(is.finite(opt$rhos) & opt$rhos >= 0 & opt$rhos < 1))
  opt$within <- sort(opt$within)
  opt$rhos <- sort(opt$rhos)
  dir.create(opt$out, recursive = TRUE, showWarnings = FALSE)
  plan <- opt[setdiff(names(opt), c("jobs", "out"))]
  plan$runner_md5 <- unname(tools::md5sum(RUNNER))
  plan$orchestrator_md5 <- unname(tools::md5sum("scripts/run_nonlinear_ld_sensitivity.R"))
  manifest <- file.path(opt$out, "plan.rds")
  if (file.exists(manifest) && !identical(readRDS(manifest), plan))
    stop("Sensitivity plan/source changed; use a fresh output directory")
  saveRDS(plan, manifest)
  jsonlite::write_json(plan, file.path(opt$out, "plan.json"), auto_unbox = TRUE,
                       pretty = TRUE, digits = 16)
  provenance <- file.path(opt$out, "provenance")
  dir.create(provenance, showWarnings = FALSE)
  for (path in c(RUNNER, "scripts/run_nonlinear_ld_sensitivity.R"))
    if (!file.copy(path, file.path(provenance, basename(path)), overwrite = TRUE))
      stop("Could not archive source: ", path)

  # Require statistical settings, implementation and package versions to match
  # before importing any legacy records. Worker count is not statistical state.
  legacy <- if (nzchar(opt$reuse)) readRDS(file.path(opt$reuse, "config.rds")) else NULL
  if (!is.null(legacy)) {
    required <- list(n = 1000L, trees = 100L, within = .6, h2 = .5,
                     background = 16L, evaluation = 32L, paths = 16L,
                     dominance_fractions = c(0, .5, 1), interaction_strengths = c(0, .5, 1, 2),
                     maf_mode = "balanced", seed_base = 2026101800L, methods = METHODS,
                     source_md5 = plan$runner_md5,
                     rf_settings = list(mtry = BLOCKS * SNPS, min_node_size = 1L, min_bucket = 5L,
                                        replace = TRUE, sample_fraction = 1, splitrule = "variance"),
                     versions = list(R = R.version.string, mvtnorm = as.character(packageVersion("mvtnorm")),
                                     ranger = as.character(packageVersion("ranger")),
                                     jsonlite = as.character(packageVersion("jsonlite"))))
    for (key in names(required))
      if (!isTRUE(all.equal(legacy[[key]], required[[key]], check.attributes = TRUE)))
        stop("Legacy run is incompatible: ", key)
    if (opt$reps > legacy$reps) cat("Reusing available legacy replicates; computing the remainder\n")
  }
  total <- length(opt$within) * length(opt$rhos) * 7L * opt$reps
  cat(sprintf("Expanded grid: %d fits, within={%s}, between={%s}\n", total,
              paste(opt$within, collapse = ","), paste(opt$rhos, collapse = ",")))
  flush.console()
  for (within in opt$within) {
    out <- file.path(opt$out, paste0("within_", gsub(".", "p", as.character(within), fixed = TRUE)))
    dir.create(file.path(out, "replicates"), recursive = TRUE, showWarnings = FALSE)
    import_path <- file.path(out, "reused_records.csv")
    if (!is.null(legacy) && within == legacy$within && !file.exists(import_path)) {
      paths <- list.files(file.path(opt$reuse, "replicates"), "\\.rds$", full.names = TRUE)
      imported <- list()
      for (path in paths) {
        record <- readRDS(path)
        row <- record$rows[1L, ]
        if (!row$rho_between %in% opt$rhos || row$replicate >= opt$reps) next
        stopifnot(identical(names(record$scores), METHODS), nrow(record$rows) == 2L,
                  basename(path) == paste0(setting_key(row$scenario, row$strength,
                                                     row$rho_between, row$replicate), ".rds"))
        target <- file.path(out, "replicates", basename(path))
        if (!file.exists(target) && !file.copy(path, target)) stop("Could not import: ", path)
        hashes <- unname(tools::md5sum(c(path, target)))
        if (hashes[1L] != hashes[2L]) stop("Imported file differs: ", target)
        imported[[length(imported) + 1L]] <- data.frame(source = path, target = target, md5 = hashes[1L])
      }
      if (!length(imported)) stop("No compatible legacy records found")
      write.csv(do.call(rbind, imported), import_path, row.names = FALSE)
      cat(sprintf("Within %.2f: imported %d byte-identical records\n", within, length(imported)))
      flush.console()
    }
    call <- c(RUNNER, "--reps", opt$reps, "--jobs", opt$jobs, "--within", within,
              "--rhos", paste(opt$rhos, collapse = ","), "--out", out)
    cat(sprintf("Running within %.2f; progress log: %s\n", within, file.path(out, "run_log.txt")))
    flush.console()
    status <- system2(file.path(R.home("bin"), "Rscript"), vapply(call, shQuote, character(1L)),
                      stdout = file.path(out, "run_log.txt"), stderr = file.path(out, "run_log.txt"))
    if (status != 0L) stop("Benchmark failed for within = ", within)
  }
  cat("All sensitivity fits completed. Run scripts/report_nonlinear_ld_sensitivity.R on this directory.\n")
}

if (sys.nframe() == 0L) main_sensitivity()
