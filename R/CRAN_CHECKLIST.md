# CRAN submission checklist: survivalGPU 0.1.0

From the audit of 6 October 2026, branch `CRAN_release`.

`R CMD check --as-cran` is already clean (0 errors, 0 warnings, 1 NOTE
"New submission"). Everything below is what the checker does not catch.

- One numbered box = one commit.
- Work top to bottom. Section 0 can run in parallel: it only blocks the
  author list in 1.3 and the submission itself.
- Items that knit vignettes or run tests or examples (2.1, 2.6, 2.7, 3.2,
  3.5, 3.6) need the `survivalGPU` Python environment. If it does not exist
  on the machine, create it first with section 3 of
  `vignettes/installation.Rmd`.
- This file is excluded from the package by `.Rbuildignore`.

## 0. Decision to make (you and Jean)

- [ ] **0.1 License of the code taken from the `WCE` package.**
  `WCE` is GPL-2 | GPL-3; this package declares LGPL (>= 2.1), and GPL code
  cannot be redistributed under LGPL terms. What was copied:
  `sumWCEall()` in [R/wceGPU.R](R/wceGPU.R#L463) (it is `WCE:::sumWCEbest()`),
  `HR()` (logic and error message of `WCE:::HR.WCE`), and three paragraphs of
  `wceGPU.Rd` (`nknots`, `constrained`, Details). Pick one:
  - a. License the R package as `GPL-2 | GPL-3` and credit WCE's authors in
    `Authors@R`.
  - b. Get written permission from WCE's authors to keep LGPL, and credit them.
  - c. Rewrite those pieces from scratch (one summary printer, one error
    message, three doc paragraphs).
  - Also judge whether `python/survivalgpu/wce_features.py` ("implements the
    core numerical routines of the WCE package") is an independent
    re-implementation or a port. Only you know how it was written.

## 1. Must fix before submitting

- [x] **1.1 Tests must not start Python on CRAN.** Done in `3baa609`.
  Line 1 of seven test files calls
  `reticulate::py_module_available("survivalgpu")`. On a machine with no Python
  configured, reticulate then downloads uv, CPython and numpy (210 MB) into
  `~/.cache/R/reticulate`. The same line also skips every test unless
  `survivalgpu` is pip-installed.
  - Add `tests/testthat/helper-backend.R` with `skip_if_no_backend()`:
    `skip_on_cran()` first, then `py_module_available("torch")`.
  - Call it on line 1 of the seven gated test files.
  - Done when: the CRAN-like run in 3.4 writes nothing to HOME, and
    `devtools::test()` with torch available runs every test file.

- [x] **1.2 A `\value` section for every documented function.** Done in
  `2f4a260`.
  - `use_cuda()`: add `@return` (one logical).
  - `print.coxphGPU()`: drop `@inherit survival::print.coxph` (it pulls in the
    `coxph.object` page), write its own description, `@return` `x` invisibly,
    and make the function return `invisible(x)`.
  - `predict.coxphGPU()`, `residuals.coxphGPU()`: add `@return`.
  - `wceGPU()`: replace "WCE results" with the class and its components.
  - `print.wceGPU()`: return `invisible(x)`.
  - Run `devtools::document()`. This also ships the `nknots` multi-knot text
    that is missing from the current `wceGPU.Rd`.
  - Done when: `grep -L '\\value' man/*.Rd` lists only
    `survivalGPU-package.Rd`.

- [x] **1.3 DESCRIPTION text and authors.** Done in `9ceda5f`, except WCE's
  authors (see 0.1).
  - [x] `Description`: method references added in CRAN's format,
    `Cox (1972) <doi:10.1111/j.2517-6161.1972.tb00899.x>` and
    `Sylvestre and Abrahamowicz (2009) <doi:10.1002/sim.3701>`.
  - [x] `Description`: "graphics processing units (GPUs)" spelled out once.
  - [x] `Authors@R`: Terry Therneau is now `role = c("ctb", "cph")`, with a
    comment saying what was adapted from `survival`. WCE's authors wait for
    0.1.
  - [x] `Depends: R (>= 4.1.0)` added (the tests use `|>`).

- [x] **1.4 Remove "not on CRAN yet" from shipped text.** Done in `9ceda5f`.
  - [x] Only the way the R package itself is installed changed. In
    [vignettes/installation.Rmd](vignettes/installation.Rmd#L18) (intro,
    sections 1 and 2) and the "Installation" section of `README.Rmd`:
    `install.packages("survivalGPU")` first, GitHub as the development
    version.
  - [x] The compiler prerequisite is narrowed, not dropped: `pykeops` compiles
    its routines at run time, so a compiler is still needed on Linux and for
    the WCE model on macOS. Windows and Cox-only use on macOS need none.
  - [x] Sections 3 and 4 kept as they are. Creating the `survivalGPU` Python
    environment is still required when it does not exist yet, whichever way
    the R package was installed. The chunks are `eval = FALSE`, so nothing is
    installed at build or check time.
  - [x] Same change mirrored by hand in `README.md` and in
    `docs/source/r/installation.md`.
  - Timing: `install.packages("survivalGPU")` only works once CRAN has
    accepted the package. Publish the README and docs site with this text at
    acceptance, or accept a short gap.

## 2. Should fix (CRAN will not notice, users will)

- [x] **2.1 Two documented examples crash.** Done in `632f6bd`.
  - [x] [R/coxphGPU.R](R/coxphGPU.R#L51): `bootstrap = 1` without
    `patient_id`, under a "without bootstrap" comment. Now `bootstrap = 0`.
  - [x] [R/wceGPU.R](R/wceGPU.R#L115): `nbootstraps = 1`. Now
    `nbootstraps = 0`.
  - Done when: `devtools::run_examples(run_dontrun = TRUE)` passes. Checked:
    all six examples run.

- [x] **2.2 `wceGPU()` bootstrap crashes.** Done in `632f6bd`.
  - [x] The fit failed with `nbootstraps = 1`, and with any bootstrap when
    there is exactly one covariate. Python returns the bootstrap coefficients
    with shape (replicates, 1, coefficients), and the `drop()` at
    [R/wceGPU.R](R/wceGPU.R#L303) collapsed every axis of length 1. Now only
    the middle axis is removed.
  - [x] `summary()` failed for a bootstrap fit without covariates. It now
    skips the covariates' bootstrap intervals in that case.
  - [x] Three tests added in `tests/testthat/test-wceGPU.R`.

- [x] **2.3 `coxphGPU()` bootstrap crashes.** Done in `632f6bd`.
  - [x] Rows with `NA` ("arguments imply differing number of rows: 228, 227")
    and `subset` (same error): the patient ids were taken from every row of
    `data`, not from the rows kept in the model frame. Fixed at
    [R/coxphGPU.R](R/coxphGPU.R#L1035).
  - [x] `bootstrap = 1`: the fit worked, but `summary()` then failed with
    "dim(X) must have a positive length". `coef_bootstrap` is now stored as
    soon as `bootstrap > 0`.
  - [x] Three tests added in `tests/testthat/test-coxphGPU.R`.

- [x] **2.4 `print()` and `plot()` for `wceGPU` objects.** Done in `bd0f7bf`.
  - [x] `print()` printed `[1] "Estimated WCE function\n:"`
    ([R/wceGPU.R](R/wceGPU.R#L380)). It now prints the plain line.
  - [x] With a single covariate, `summary()` printed its bootstrap interval
    under the row label `[1,]`. It now shows the covariate's name.
  - [x] `plot()` titled a fit without bootstrap "with confidence interval
    (0 bootstraps)". The plain plot, with the AIC or BIC in its title, is now
    chosen from `is_bootstraps` rather than from `nbootstraps == 1`.
  - [x] `plot(hist.covariates = TRUE)` drew the point estimate, not the
    bootstrap coefficients. It now reads `bootstrap_beta.hat.covariates`.
  - [x] The confidence band ran off the plot: the y-axis was sized for the
    estimate alone. It now covers the band.
  - Checked by rendering the plots; no automated test added.

- [x] **2.5 The `predict.coxphGPU` example changes a global option.** Done in `632f6bd`.
  `options(na.action = na.exclude)` was never restored.
  - [x] The example now saves the old value and resets it at the end.

- [x] **2.6 Vignette figures are deleted at build time.** Done in `5c348bc`.
  `vignettes/precompile.R` saves figures in `<name>_files/`, which
  `rmarkdown::render()` removes. The tarball ships `coxPH.Rmd` without its
  PNGs, and knitting in place deletes the committed copies.
  - [x] `precompile.R` now uses `fig.path = "figures/<name>-"`.
  - [x] The three existing PNGs moved to `vignettes/figures/` under the names
    that setting produces, and `coxPH.Rmd` points at them. Nothing was
    re-knit, so no number in the vignette changed.
  - [x] `precompile.R` was not tracked: `vignettes/.gitignore` ignores `*.R`.
    An exception `!precompile.R` is added; the script still has to be
    `git add`-ed once.
  - Done when: `tar tzf survivalGPU_0.1.0.tar.gz | grep vignettes/` lists the
    PNGs. Checked: it does, and rendering in place leaves them alone.

- [x] **2.7 Vignette sources and results.** Done in `5c348bc`.
  - [x] Titles: the only prose that differed between each `.Rmd` and its
    `.orig`. Both pairs now say "Cox Proportional Hazards" and "Weighted
    Cumulative Exposure (WCE)" (the most recent edit of each pair).
  - [x] `WCE.Rmd.orig`: the six `eval=FALSE` chunks are evaluated. The
    vignette now shows the fit, both summaries, the knot counts, the plot
    (`figures/WCE-wceGPU_plot-1.png`) and the hazard ratio.
  - [x] Both vignettes re-knit with the `survivalGPU` virtualenv. The Cox
    figures are byte-identical; its bootstrap interval moved slightly, as the
    bootstrap has no seed.
  - [x] The `[KeOps] Warning : CUDA libraries not found` line is kept out of
    the docs only: `precompile.R` sets `KEOPS_VERBOSE=0`, and the line is
    removed from `README.md`. The package and its messages are unchanged.
  - [x] `summary()` printed "Use plot(object)": a leftover line in
    `sumWCEall()` overwrote the object's name. Removed.
  - [x] `precompile.R` now refers to `vignette("installation")`.

- [x] **2.8 The "no Python" error never names the cause.** Done in `21cf508`.
  With torch missing, `survivalgpu_unavailable_error()` in `R/zzz.R` reported
  "Error loading Python module survivalgpu", then
  "ValueError: list.remove(x): x not in list" on every later call.
  - [x] It now says when Python itself cannot be started.
  - [x] Otherwise it lists exactly the Python packages that are missing, and
    the Python in use. `pykeops` is checked only when called from `wceGPU()`.
  - [x] If nothing is missing, the original Python error is passed through.
    Before, any error during a WCE fit (a wrong column name, for example) was
    reported as a missing Python installation.
  - Checked by hand in five purpose-built environments; no automated test.

- [ ] **2.9 DESCRIPTION and package-doc hygiene.**
  - Drop `pkgdown` from `Suggests`.
  - `SystemRequirements`: also name numpy, pandas, scipy, matplotlib, beartype
    and jaxtyping, and the Python version (>= 3.10).
  - Add `graphics` to `Imports` (NAMESPACE imports from it; the check does not
    flag this).
  - Remove the `@author` block in `R/survivalGPU-package.R`: it omits the
    maintainer, and roxygen fills it from `Authors@R` otherwise.

- [ ] **2.10 Unused and unregistered C code.**
  - Used: `agmart3.c` and `coxmart.c` (martingale residuals, on every fit),
    and `coxcount1.c` (only for formulas with a `tt()` term).
  - [x] Deleted the four files that nothing called (`chinv2.c`,
    `cholesky2.c`, `chsolve2.c`, `dmatrix.c`) and their prototypes in
    `survproto.h`.
  - [ ] `multicheck.c` and `tmerge.c` are kept for now, on purpose.
    `multicheck` and `tmerge3` are called from `R/coxph_internals.R` but not
    registered in `src/init.c`, so the calls fail if reached. Still to decide:
    delete the multi-state code that can never run (`multi` is always
    `FALSE`), and either register `tmerge3` or reject `Surv2` responses up
    front.

- [ ] **2.11 The pykeops install hint is wrong for R users.**
  `pip install survivalgpu[wce]` appears in `tests/testthat/helper-keops.R`
  and in the Python error messages (`__init__.py`, `wce_features.py`). R users
  need `reticulate::virtualenv_install("survivalGPU", "pykeops")`.

- [ ] **2.12 Make the "no-python" CI job behave like CRAN.**
  In `../.github/workflows/R-CMD-check.yaml`, job `R-CMD-check-no-python`: add
  `NOT_CRAN: "false"` to `env`. `setup-r` sets `NOT_CRAN=true` otherwise, so
  `skip_on_cran()` would not apply there.

- [ ] **2.13 Optional cleanup.**
  - Four PNGs in `man/figures/` are no longer referenced (170 KB):
    `README-unnamed-chunk-5-1`, `-6-1`, `-7-1` and `README-wceGPU_plot-1`.
  - `spelling::update_wordlist()`.

- [ ] **2.14 Small inaccuracies in the help pages.**
  - `wceGPU.Rd`: `allres` is documented as "returns linear predictors,
    wald.test, concordance for all bootstraps", but `summary(allres = TRUE)`
    prints the table of candidate `nknots`.
  - `wceGPU.Rd`: the description contains a literal "(see @details)".
  - `coxphGPU.Rd` and `wceGPU.Rd`: the bootstrap arguments are described as
    "bootstrap cross-validation"; it is bootstrap resampling.

## 3. Final verification (after everything above is committed)

- [ ] **3.1** `devtools::document()` leaves `git status` clean.
- [ ] **3.2** `Rscript vignettes/precompile.R` and `devtools::build_readme()`
  leave `git status` clean.
- [ ] **3.3** Delete the old `survivalGPU_0.1.0.tar.gz` (17 September) and build
  a fresh one from the clean tree: `R CMD build .`
- [ ] **3.4** CRAN-like check: no Python configured, empty HOME.

  ```sh
  LIB=$(Rscript -e 'cat(.libPaths()[1])')
  mkdir -p /tmp/cran-sim/home && cp survivalGPU_0.1.0.tar.gz /tmp/cran-sim/ && cd /tmp/cran-sim
  env -i HOME=$PWD/home PATH=/usr/local/bin:/usr/bin:/bin LANG=en_US.UTF-8 \
    R_LIBS_USER=$LIB _R_CHECK_CRAN_INCOMING_USE_ASPELL_=true \
    R CMD check --as-cran survivalGPU_0.1.0.tar.gz
  find home -mindepth 1 | head   # must print nothing
  ```

  Expected: 0 errors, 0 warnings, 1 NOTE (new submission, plus six words as
  possibly misspelled: "Abrahamowicz", "Sylvestre", "GPUs", "Scalable", "WCE"
  and "natively").
- [ ] **3.5** With the `survivalGPU` virtualenv: `devtools::test()` has no
  failure and no skip apart from "empty test".
- [ ] **3.6** `devtools::run_examples(run_dontrun = TRUE)` passes.
- [ ] **3.7** `urlchecker::url_check()` and `spelling::spell_check_package()`.
- [ ] **3.8** `devtools::check_win_devel()`, `devtools::check_win_release()`
  and `devtools::check_mac_release()`. These upload the tarball to external
  build services.
- [ ] **3.9** Rewrite `cran-comments.md` from the results of 3.4 and 3.8:
  - remove the "invalid URLs" paragraph (the NOTE no longer occurs);
  - list the six flagged words as correctly spelled (two are author names);
  - say the tests are skipped on CRAN with `skip_on_cran()`;
  - keep the justification for `\dontrun{}`.

## 4. Submit

- [ ] **4.1** Upload the tarball from 3.3 at
  <https://cran.r-project.org/submit.html> (or `devtools::submit_cran()`),
  then confirm through the email sent to the maintainer address.
- [ ] **4.2** After acceptance: tag the commit, set `Version: 0.1.0.9000`, and
  add a development heading to `NEWS.md`. Publishing a GitHub *release*
  triggers `publish-pypi.yml` and the docs deployment, so decide whether you
  want that before creating one.
