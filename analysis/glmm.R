# H1a GLMM (prereg-v1, analysis_plan.md §4). Usage: Rscript analysis/glmm.R <input.csv> <output.csv>
# input columns: dataset, lang, sentence_id, correct (0/1), overlap_ind
suppressPackageStartupMessages(library(lme4))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1], colClasses = c(sentence_id = "character"))
out <- list()
for (ds in unique(d$dataset)) {
  x <- d[d$dataset == ds, ]
  x$overlap_z <- as.numeric(scale(x$overlap_ind))
  x$sentence_id <- factor(x$sentence_id); x$lang <- factor(x$lang)
  ctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  ok <- function(m) is.null(m$message) && !isSingular(m$fit) &&
    length(m$fit@optinfo$conv$lme4$messages) == 0
  fit1 <- function(f) tryCatch(list(fit = glmer(f, data = x, family = binomial, control = ctl), message = NULL),
                               error = function(e) list(fit = NULL, message = conditionMessage(e)))
  used <- "glmer(correct ~ overlap_z + (1|sentence_id) + (1|lang))"
  m <- fit1(correct ~ overlap_z + (1 | sentence_id) + (1 | lang))
  if (!ok(m)) { used <- "glmer(correct ~ overlap_z + (1|sentence_id))"; m <- fit1(correct ~ overlap_z + (1 | sentence_id)) }
  if (!is.null(m$fit) && ok(m)) {
    b <- fixef(m$fit)[["overlap_z"]]
    ci <- tryCatch(confint(m$fit, parm = "overlap_z", method = "profile"), error = function(e) confint(m$fit, parm = "overlap_z", method = "Wald"))
    out[[ds]] <- data.frame(dataset = ds, model = used, beta = b, ci_lo = ci[1], ci_hi = ci[2], n = nrow(x),
                            n_lang = nlevels(x$lang), converged = TRUE)
  } else {  # pre-declared fallback 2: GEE is run in Python (analysis/run_confirmatory.py)
    out[[ds]] <- data.frame(dataset = ds, model = "FALLBACK_GEE", beta = NA, ci_lo = NA, ci_hi = NA, n = nrow(x),
                            n_lang = nlevels(x$lang), converged = FALSE)
  }
}
write.csv(do.call(rbind, out), args[2], row.names = FALSE)
cat(sprintf("R %s, lme4 %s\n", getRversion(), packageVersion("lme4")))
