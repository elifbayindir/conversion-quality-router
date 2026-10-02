# Presentation Outline (15 minutes)

Maps a standard 15-minute technical walkthrough to inspectable, verifiable
evidence already committed in this repository. Each section names the exact
file(s) a reviewer can open to check the claim made in that section — no
section here asserts anything that isn't backed by a specific path below.

| # | Time | Section | Repository evidence |
|---|---|---|---|
| 1 | 0:00–1:30 | Problem statement and scope | [`README.md`](../README.md#problem-statement), [`README.md`](../README.md#honest-project-scope-and-non-claims) |
| 2 | 1:30–3:00 | Dataset, leakage handling, feature contract | [`data/README.md`](../data/README.md), [`docs/leakage-audit.md`](leakage-audit.md), [`artifacts/metadata/feature_contract.json`](../artifacts/metadata/feature_contract.json) |
| 3 | 3:00–4:30 | Split design and duplicate-group fix | [`artifacts/metadata/split_summary.json`](../artifacts/metadata/split_summary.json), [`docs/leakage-audit.md`](leakage-audit.md#duplicate-row-split-leakage-found-and-fixed) |
| 4 | 4:30–6:30 | Model: baseline vs. MLP, calibration, threshold | [`docs/model-card.md`](model-card.md), [`docs/baseline-report.md`](baseline-report.md), [`artifacts/metadata/mlp_metadata.json`](../artifacts/metadata/mlp_metadata.json) |
| 5 | 6:30–8:00 | Final test metrics and error analysis | [`docs/test-evaluation-report.md`](test-evaluation-report.md), [`artifacts/metadata/test_evaluation.json`](../artifacts/metadata/test_evaluation.json) |
| 6 | 8:00–9:30 | Decision agent: guardrails, schema, fallback | [`docs/agent-card.md`](agent-card.md), [`docs/agent-policy.md`](agent-policy.md), [`prompts/decision_agent_v1.md`](../prompts/decision_agent_v1.md) |
| 7 | 9:30–11:00 | n8n automation: branches, approval, error handling | [`docs/n8n-workflow.md`](n8n-workflow.md), [`automation/conversion_quality_router.json`](../automation/conversion_quality_router.json), [`docs/automation-test-report.md`](automation-test-report.md) |
| 8 | 11:00–12:30 | Live demo | [`docs/demo-script.md`](demo-script.md) (this is the ~2-minute recorded segment, embedded here) |
| 9 | 12:30–13:30 | QA: acceptance matrix, clean-room, security audit | [`docs/qa-report.md`](qa-report.md), [`docs/qa_matrix_results.json`](qa_matrix_results.json), [`docs/evaluation-card.md`](evaluation-card.md) |
| 10 | 13:30–14:30 | Known limitations and responsible use | [`README.md`](../README.md#known-limitations), [`README.md`](../README.md#responsible-use-non-use-cases), [`docs/evaluation-card.md`](evaluation-card.md#limitations-and-next-evaluation-steps) |
| 11 | 14:30–15:00 | Q&A | [`docs/q-and-a.md`](q-and-a.md) |

## Notes for the presenter

- Every number quoted in sections 4-5 should be read live from
  `artifacts/metadata/*.json` or the corresponding report on screen, not
  recited from memory — this is the same discipline the cards themselves
  follow (see the cross-check note in
  [`docs/evaluation-card.md`](evaluation-card.md)).
- Section 8 (live demo) is the pre-recorded segment from
  [`docs/demo-script.md`](demo-script.md); it does not need to be re-run
  live if time is short, but the underlying commands are all real and
  re-runnable if a reviewer asks to see it live instead.
- If time runs short, sections 2-3 (dataset/leakage/split) can be
  compressed to a single slide pointing at
  [`docs/leakage-audit.md`](leakage-audit.md) — the QA section (9) and
  limitations section (10) should not be cut, since they are what
  distinguishes a demonstrated result from an honestly-scoped one.
