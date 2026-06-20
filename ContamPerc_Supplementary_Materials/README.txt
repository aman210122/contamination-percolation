ContamPerc: Supplementary Materials
Manuscript: "Contamination Percolation in Multi-Agent LLM Systems: A Measurement
Framework and Benchmark" (IEEE Access, Access-2026-13105)
Author: Aman Sharma

These materials accompany the manuscript and are also available in the public
repository: https://github.com/aman210122/contamination-percolation
(commit 8509140cfb9327048a97c65b616ab82057867509).

Note on model labeling: some raw files use the endpoint name "dbrx" for the model
identified in the manuscript as GPT-OSS 120B (served through Databricks Mosaic AI).
This is the same endpoint and the same data; the manuscript corrects the display
label. No numbers change.

----------------------------------------------------------------------
S1_detector_validation/  (referenced in the Detector Validation section)
----------------------------------------------------------------------
- detector_labeling_workbook.xlsx : the physician's labeling workbook (150 agent
  outputs labeled DEFENSIVE / PROPAGATED / AMBIGUOUS by Dr. Muhammad Bhatty),
  with the auto-computed Summary sheet.
- labeled_outputs_full.csv        : the full labeled dataset exported to CSV
  (model, topology, tau, agent role, trial, output text, detector prediction,
  physician label, notes).
- confusion_matrix_nonambiguous.csv : confusion matrix on the 133 non-ambiguous
  outputs (physician label x detector prediction). Reproduces precision 0.98,
  recall 0.95 (F1 0.96) reported in the manuscript.
- agreement_by_model.csv          : per-model detector-physician agreement.
- agreement_by_agent_role.csv     : per-pattern (agent role) agreement breakdown.

----------------------------------------------------------------------
S2_raw_experiment_data/  (per-cell raw outputs underlying all tables and figures)
----------------------------------------------------------------------
- E1_phase_transition.json        : canary phase-transition sweep (all models).
- E2_tier1_semantic*.json         : Tier-1 semantic experiment, overall and per
  model, including E2_tier1_semantic_gpt_oss_20b.json (the within-family size
  control) and E2_tier1_semantic_gpt_oss_120b.json.
- E6_tier1_control_baseline.json  : no-injection baseline control.
- canary_results.json, tier1_results.json : aggregated MIMIC-IV real-case canary and Tier-1 results for the five evaluated models (GPT-OSS 120B [dbrx], Claude, Llama, Gemini 2.5 Flash, GPT-4o-mini).
- cross_domain_results_*.json     : legal and financial cross-domain runs (all
  five models across the two files).
- sensitivity_results*.json       : temperature and prompt-phrasing sensitivity,
  overall and per model.
- summary.json                    : run summary.

----------------------------------------------------------------------
S3_scripts/  (agent protocol and the verbatim agent system prompts)
----------------------------------------------------------------------
- run_experiments.py    : core canary and Tier-1 harness; contains the five
  clinical agent system prompts (triage, diagnostic, treatment, pharmacy, safety
  monitor) and the legal/financial role templates. The Reproducibility appendix
  of the manuscript points here for the agent prompts.
- run_mimic_experiment.py : MIMIC-IV real-case slice harness, including the
  marker auto-selection used for the physician audit. The per-case marker
  selections and the five physician-flagged replacement suggestions are recorded
  in the labeling workflow and the repository.
- run_cross_domain.py   : cross-domain (legal, financial) harness.
- run_sensitivity.py    : temperature and prompt-phrasing sensitivity harness.

The exact hyperparameters (temperature 0.3, max_tokens 512), random seed handling,
the 13 tau values, and dependency versions are listed in the Reproducibility
appendix of the manuscript.
