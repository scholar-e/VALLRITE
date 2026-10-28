# Phone candidate deletion audit, 2026-10-28

After fixed ranking reached 93.29% WER with the mixed decoder, this diagnostic
examined candidate quality before authorizing longer training. All results use
the repeatedly inspected development set, not an independent test set.

## Saved-candidate audit

`audit_phone_candidates.py` measures minimum-edit substitutions, deletions and
insertions, length ratios and five-candidate oracle PER. Alignment ties prefer
diagonal operations then deletion; individual operation counts are not unique
in every ambiguous alignment. Every row is checked against independent edit
distance and the insertion/deletion length identity. Reports include source
hashes and reference-length bins.

The original development export contains 44,586 reference phones but only
10,120 rank-one predicted phones (22.70%). Of 36,616 errors, 34,486 are deletions
(94.18%). In 1,053 of 1,140 clips, predictions are shorter than half the
reference. Even summing the longest candidate per clip yields only 11,428
phones. This explains why sentence selection cannot recover most missing words.

The 1,500-clip training export also exhibits severe underproduction: 18,668
reference phones versus 5,912 predicted phones, with 12,759 deletions, 1,137
substitutions and three insertions. This export is in-sample for the pretrained
encoder and biased toward the first eligible clips; its PER is not a held-out
accuracy estimate.

## Data checks and local review

All 1,140 development IDs and phone references match the validation manifest.
None has frame_start/frame_end slicing or a pretrain chunk index. Decoded frame
counts match manifest counts for the five shortest and five longest eligible
clips. This spot check does not establish correct audiovisual timing or validate
every source transcript.

Local DeepSeek performed a read-only review (job
`3c6aba26-5bcc-4f00-89c0-c44bc6efb070`). Its proposed landmark truncation issue
does not apply to this image-only path. Its claim that capping added endpoint
padding removes real source frames is unsupported. Neither was treated as a
confirmed defect. No production data-loader changes were made.

## Removing the phone language-model prior

One bounded ablation changes phone LM weight from 0.5 to zero. The checkpoint
remains `fullclip-phone-context-20261013/phones-step-0750.pt`; beam width 64,
token top-k 12, token bonus zero and five exported candidates are unchanged.
No weight sweep or reference-based length tuning is performed. GPU encoding is
followed by CPU beam search, wrapped in thermal protection and a ten-minute
process cap. Evaluation completed in 258.63 seconds.

| Metric | Phone LM weight 0.5 | No phone LM |
|---|---:|---:|
| Rank-one beam PER | 82.12% | 70.82% |
| Oracle PER@5 | 79.96% | 69.29% |
| Deletions | 34,486 | 22,575 |
| Substitutions | 2,110 | 8,848 |
| Insertions | 20 | 153 |
| Predicted/reference phone count | 22.70% | 49.71% |
| Empty rank-one candidates | 16 | 0 |

Greedy PER remains exactly 82.6717%, confirming unchanged greedy inference.
The no-prior beam removes 5,040 total rank-one errors and 4,757 oracle errors.
Removing the prior reduces deletions substantially, but the remaining error is
still large. These numbers do not establish CTC blank occupancy; frame logits
would be needed to measure that directly.

Artifacts: `checkpoints/phone-candidate-audit-{dev,train,no-lm}-20261028` and
`checkpoints/context-no-phone-lm-20261028`. Reproduce an audit with
`python -m CompetitiveWordDecoder.audit_phone_candidates --source EXPORT.jsonl
--output NEW_DIRECTORY` in the project environment.

## Frozen word-decoder evaluation

The unchanged mixed adapter decoded all 5,700 candidates, followed by the same
fixed summed-log-probability ranking rule. Prompt, generation settings and
normalization were unchanged. Each stage completed within its ten-minute cap;
generation took 252.11 seconds and ranking about 45 seconds. Thermal protection
paused ranking briefly for CPU cooling and resumed automatically.

| Mixed decoder metric | Phone LM weight 0.5 | No phone LM |
|---|---:|---:|
| Unranked WER | 95.23% (12,070 errors) | 101.69% (12,889 errors) |
| Fixed-ranked WER | 93.29% (11,824 errors) | 93.33% (11,829 errors) |
| Oracle WER@5 | 90.12% (11,423 errors) | 90.26% (11,441 errors) |
| Capped generations | 7 | 19 |

Ranking preserved every candidate and the original oracle count. The five
ranker tests pass. Its manifest caveat now says path scores *may* include a
phone prior rather than incorrectly hardcoding weight 0.5 for every export;
the scoring algorithm itself is unchanged.

Decision: do not replace the current ranked pipeline. Better phone PER did not
improve WER with this frozen decoder; ranked WER differs by only five errors,
while unranked output worsens substantially. The mixed adapter was trained on
the old prior-weighted errors, so distribution mismatch is a plausible cause,
not a demonstrated explanation. The result does not justify more visual-head
training yet. Next proposed comparison: test the retained clean EOS decoder on
these same no-prior candidates before training a decoder on newly matched
training-side errors. Keep the existing phone exports and word baselines for
matched comparisons. No new training was launched in this audit.
