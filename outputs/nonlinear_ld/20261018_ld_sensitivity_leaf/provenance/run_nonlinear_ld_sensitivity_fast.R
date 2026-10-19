#!/usr/bin/env Rscript
source("scripts/run_nonlinear_ld.R")
source("scripts/nonlinear_ld_leaf_shap.R")

main_fast_sensitivity <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opt <- list(reps = 100L, jobs = 12L,
              out = "outputs/nonlinear_ld/20261018_ld_sensitivity_leaf",
              reuse = "outputs/nonlinear_ld/20261018_ld_sensitivity")
  if (length(args) %% 2L) stop("Use --name value pairs")
  if (length(args)) for (i in seq.int(1L, length(args), 2L)) {
    key <- sub("^--", "", args[i])
    if (!key %in% names(opt)) stop("Unknown option: ", key)
    opt[[key]] <- if (key %in% c("out", "reuse")) args[i + 1L] else as.integer(args[i + 1L])
  }
  stopifnot(opt$reps >= 2L, opt$jobs >= 1L)
  within <- c(.6, .9, .99)
  rhos <- c(0, .3, .6, .9, .99)
  sources <- c(RUNNER, "scripts/nonlinear_ld_leaf_shap.R", "scripts/nonlinear_ld_leaf_shap.c",
               "scripts/run_nonlinear_ld_sensitivity_fast.R")
  backend <- list(name = "tree-leaf evaluation of the unchanged grouped game",
                  source_md5 = as.list(tools::md5sum(sources)))
  equivalence_path <- "/tmp/nonlinear_ld_leaf_equivalence.csv"
  equivalence <- read.csv(equivalence_path)
  stopifnot(nrow(equivalence) == 12L, max(equivalence$maximum_phi_error) < 1e-10,
            file.info(equivalence_path)$mtime >= max(file.info(sources[2:3])$mtime))
  base <- readRDS("outputs/nonlinear_ld/20261017_r_mvtnorm/config.rds")
  stopifnot(identical(base$source_md5, unname(tools::md5sum(RUNNER))))
  versions <- list(R = R.version.string, mvtnorm = as.character(packageVersion("mvtnorm")),
                   ranger = as.character(packageVersion("ranger")), jsonlite = as.character(packageVersion("jsonlite")))
  stopifnot(identical(base$versions, versions))
  dir.create(opt$out, recursive = TRUE, showWarnings = FALSE)
  plan <- list(reps = opt$reps, within = within, rhos = rhos, reuse = opt$reuse,
               runner_md5 = base$source_md5, backend = backend)
  manifest <- file.path(opt$out, "plan.rds")
  if (file.exists(manifest) && !identical(readRDS(manifest), plan)) stop("Plan/source changed; use a fresh output directory")
  saveRDS(plan, manifest)
  jsonlite::write_json(plan, file.path(opt$out, "plan.json"), auto_unbox = TRUE, pretty = TRUE, digits = 16)
  provenance <- file.path(opt$out, "provenance")
  dir.create(provenance, showWarnings = FALSE)
  for (path in sources) stopifnot(file.copy(path, file.path(provenance, basename(path)), overwrite = TRUE))
  stopifnot(file.copy(equivalence_path, file.path(opt$out, "backend_equivalence.csv"), overwrite = TRUE))
  load_leaf_backend(file.path(opt$out, "native_backend"))
  # The original run_one closure exposes its ranger model to this adapter.
  # Its inputs, random group orders and output schema remain unchanged.
  group_shap <<- function(predict_fun, background, evaluation, paths = 16L, seed = 1L, exact = FALSE) {
    model <- get("model", envir = environment(predict_fun), inherits = FALSE)
    group_shap_leaf(model, background, evaluation, paths, seed, exact)
  }
  for (w in within) {
    name <- paste0("within_", gsub(".", "p", as.character(w), fixed = TRUE))
    out <- file.path(opt$out, name)
    dir.create(file.path(out, "replicates"), recursive = TRUE, showWarnings = FALSE)
    config <- base
    config$reps <- as.numeric(opt$reps)
    config$rhos <- rhos
    config$within <- w
    config$backend <- backend
    config_manifest <- file.path(out, "config.rds")
    if (file.exists(config_manifest) && !identical(readRDS(config_manifest), config)) stop("Config changed: ", out)
    saveRDS(config, config_manifest)
    jsonlite::write_json(config, file.path(out, "config.json"), auto_unbox = TRUE, pretty = TRUE, digits = 16)
    writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
    legacy_path <- file.path(opt$reuse, name)
    import_path <- file.path(out, "reused_records.csv")
    if (dir.exists(legacy_path) && !file.exists(import_path)) {
      legacy <- readRDS(file.path(legacy_path, "config.rds"))
      keys <- setdiff(names(base), c("reps", "rhos", "within"))
      for (key in keys) stopifnot(isTRUE(all.equal(legacy[[key]], base[[key]])))
      stopifnot(legacy$within == w)
      imported <- list()
      for (path in list.files(file.path(legacy_path, "replicates"), "\\.rds$", full.names = TRUE)) {
        record <- readRDS(path)
        row <- record$rows[1L, ]
        if (row$replicate >= opt$reps || !row$rho_between %in% rhos) next
        stopifnot(identical(names(record$scores), METHODS), nrow(record$rows) == 2L,
                  basename(path) == paste0(setting_key(row$scenario, row$strength, row$rho_between, row$replicate), ".rds"))
        target <- file.path(out, "replicates", basename(path))
        if (!file.exists(target)) stopifnot(file.copy(path, target))
        hashes <- unname(tools::md5sum(c(path, target)))
        stopifnot(hashes[1L] == hashes[2L])
        imported[[length(imported) + 1L]] <- data.frame(source = path, target = target, md5 = hashes[1L])
      }
      if (length(imported)) write.csv(do.call(rbind, imported), import_path, row.names = FALSE)
      cat(sprintf("Within %g: imported %d completed reference fits\n", w, length(imported)))
      flush.console()
    }
    settings <- rbind(expand.grid(scenario = 4L, strength = config$dominance_fractions, rho = rhos,
                                  replicate = seq_len(opt$reps) - 1L),
                      expand.grid(scenario = 5L, strength = config$interaction_strengths, rho = rhos,
                                  replicate = seq_len(opt$reps) - 1L))
    tasks <- lapply(seq_len(nrow(settings)), function(i) as.list(settings[i, ]))
    missing <- !vapply(tasks, function(t) file.exists(file.path(out, "replicates",
      paste0(setting_key(t$scenario, t$strength, t$rho, t$replicate), ".rds"))), logical(1L))
    pending <- tasks[missing]
    started <- proc.time()[[3L]]
    cat(sprintf("Within %g: computing %d remaining fits on %d workers\n", w, length(pending), opt$jobs))
    flush.console()
    if (length(pending)) for (start in seq.int(1L, length(pending), by = opt$jobs)) {
      batch <- pending[seq.int(start, min(start + opt$jobs - 1L, length(pending)))]
      result <- parallel::mclapply(batch, run_one, config = config, out = out,
                                   mc.cores = opt$jobs, mc.set.seed = FALSE, mc.preschedule = FALSE)
      failed <- vapply(result, inherits, logical(1L), "try-error")
      if (any(failed)) stop(paste(result[failed], collapse = "\n"))
      cat(sprintf("Within %g: %d/%d new fits; %.1fs\n", w, min(start + opt$jobs - 1L, length(pending)),
                  length(pending), proc.time()[[3L]] - started))
      flush.console()
    }
    records <- lapply(tasks, run_one, config = config, out = out)
    summarize_records(records, out)
  }
  cat(sprintf("All %d fits are available. Run the sensitivity reporter to audit and summarize.\n",
              length(within) * length(rhos) * 7L * opt$reps))
}

if (sys.nframe() == 0L) main_fast_sensitivity()
