# PROTOCOL — PBC generative augmentation study

This file freezes experimental decisions **before** the experiments they govern are run.
Its git history is the timestamp: a decision committed before a result cannot have been
chosen because of that result. Sections are added as each step is reached; a section
marked *not yet frozen* has no binding content.

Changes after freezing are allowed only as dated amendments at the bottom, each with
its reason. Nothing above an amendment is edited.

---

## 1. Data — frozen 6 November 2026

Evidence for every decision below: `00_data_audit.py`.

**Dataset.** PBC (Acevedo et al. 2020), Barcelona release, pre-split into
`train / valid / test` folders: 14,441 / 1,605 / 845 images, 8 classes, 360×363 RGB JPEG.
Origin of the split: provided by the project supervisor; the procedure used to create it
is not documented. The audit shows it is stratified by class and was made image by image.

**D1. Evaluation sets are never modified.** Valid and test are used exactly as received,
in every condition, never subsampled, never cleaned, never augmented.

**D2. Duplicate rule.** Exact duplicates (identical md5) are handled by kind:
- label conflict, or any copy outside train → every train copy removed;
- same-class duplicates entirely inside train → first copy (alphabetical) kept, rest removed;
- groups with no train copy → kept, recorded as a known flaw.

Result: 17 training images excluded, listed in `excluded.txt`; train = 14,424.
Known flaw: one `ig` (`pmy`) image appears in both valid and test.

**D3. Rare classes.** Rule: the two smallest classes in the cleaned training set.
Result: **basophil** (1035) and **lymphocyte** (1038). Both are single-subtype.

**D4. Resolution.** Images are centre-cropped to 360×360 and resized to **128×128**
(antialiased). Every image used by any classifier, real or synthetic, passes through
128×128, so real and synthetic images are on equal footing.

Reason: visual inspection (audit, section 7) shows that at 64×64 the granule texture
that identifies basophils, one of the two rare classes, is largely lost; at 128×128 it
is still visible.

Fallback: if, after its full training budget, a generator's rare-class samples
are classified as their intended class less than 50% of the time by the baseline
classifier, all arms move to 64×64 and the switch is reported.

The cost of the bottleneck is measured by training the baseline classifier at native
resolution and at 128×128; both numbers are reported.

**D5. Test set usage.** Each final model is evaluated on test once, at the end.
Model selection, early stopping and hyperparameter choices use valid only. No decision
is made after seeing a test score.

**D6. Statistical precision.** Rare classes have 61 test images each, so the standard
error of a single recall measurement is about ±4 points. No conclusion is drawn from a
single seed.

**Prior use of the test set.** Before this protocol was written, an exploratory
ImageNet-pretrained ResNet-18 was trained on this data (224×224, selection on valid)
and evaluated once on the test set. No decision in this protocol was derived from that
evaluation, and that model is not part of the study.

---

## 2. Induced imbalance — *not yet frozen*

Subsampling levels N for the rare classes; subsampling procedure (seeded, deterministic);
whether valid is subsampled in proportion.

## 3. Seeds — *not yet frozen*

## 4. Downstream classifier — *not yet frozen*

Architecture, optimiser, schedule, augmentation, epochs: identical across every arm.

## 5. Baseline arm — *not yet frozen*

Imbalance remedies included; tuning budget (must equal that of the synthetic arms).

## 6. Generative arms — *not yet frozen*

Models, training budget, number of synthetic images added per rare class.

## 7. Metrics and decision rules — *not yet frozen*

Macro-F1, per-class recall, balanced accuracy, MCC. Effect size counted as "helped",
statistical test, multiple-comparison correction.

## 8. Generalisation and memorisation checks — *not yet frozen*

Near-duplicate leakage check (train ↔ test, feature space); nearest-neighbour analysis of
synthetic images; membership inference per arm.

---

## Amendments

_None._
