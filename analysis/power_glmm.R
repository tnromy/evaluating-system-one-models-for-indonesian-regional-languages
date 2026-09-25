# Power of H1a (prereg-v1) by parametric simulation from the VALIDATION pilot fit.
# For each dataset: fit glmer(correct ~ overlap_z + (1|sentence_id) + (1|lang)) on the pilot, then simulate test-size
# data (test sentences x same regional languages; overlap_z resampled per language from the pilot) with beta scaled
# by f in {1, 0.5}, refit, and count one-sided Wald p < 0.05 for beta > 0.
# Usage: Rscript analysis/power_glmm.R results/confirmatory_valid/h1a_input.csv prereg/power_glmm.csv
suppressPackageStartupMessages(library(lme4))
set.seed(20260924)
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1], colClasses = c(sentence_id = "character"))
n_test <- c(nusax_senti = 400, nt_senti = 1200, nt_emot = 1000)
n_sim <- 200
ctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
rows <- list()
for (ds in names(n_test)) {
  x <- d[d$dataset == ds, ]
  x$overlap_z <- as.numeric(scale(x$overlap_ind))
  fit <- glmer(correct ~ overlap_z + (1 | sentence_id) + (1 | lang), data = x, family = binomial, control = ctl)
  b0 <- fixef(fit)[[1]]; b1 <- fixef(fit)[[2]]
  vc <- as.data.frame(VarCorr(fit)); sd_s <- vc$sdcor[vc$grp == "sentence_id"]; sd_l <- vc$sdcor[vc$grp == "lang"]
  langs <- unique(x$lang); lang_eff <- ranef(fit)$lang[langs, 1]; names(lang_eff) <- langs
  for (f in c(1, 0.5)) {
    hits <- 0; ok <- 0
    for (s in seq_len(n_sim)) {
      sid <- rep(seq_len(n_test[[ds]]), times = length(langs)); lg <- rep(langs, each = n_test[[ds]])
      oz <- unlist(lapply(langs, function(l) sample(x$overlap_z[x$lang == l], n_test[[ds]], replace = TRUE)))
      eta <- b0 + f * b1 * oz + rnorm(n_test[[ds]], 0, sd_s)[sid] + lang_eff[lg]
      sim <- data.frame(correct = rbinom(length(eta), 1, plogis(eta)), overlap_z = oz, sentence_id = factor(sid), lang = factor(lg))
      m <- tryCatch(glmer(correct ~ overlap_z + (1 | sentence_id) + (1 | lang), data = sim, family = binomial, control = ctl),
                    error = function(e) NULL)
      if (is.null(m)) next
      ok <- ok + 1
      z <- coef(summary(m))["overlap_z", "z value"]
      if (z > qnorm(0.95)) hits <- hits + 1
    }
    rows[[length(rows) + 1]] <- data.frame(dataset = ds, pilot_beta = b1, effect_scale = f, test_sentences = n_test[[ds]],
                                           n_lang = length(langs), sims_ok = ok, power = hits / ok)
    cat(ds, f, hits / ok, "\n")
  }
}
write.csv(do.call(rbind, rows), args[2], row.names = FALSE)
