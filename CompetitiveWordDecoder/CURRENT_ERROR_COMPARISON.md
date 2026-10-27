# Current-model errors versus clean domain adaptation

Started2026-10-27. Controlled decoder-only pilot; visual weights remain frozen.

The context checkpoint `fullclip-phone-context-20261013/phones-step-0750.pt`
generated five-best candidates for the first1,500 eligible training clips, using
the same beam64/top-k12/bigram weight0.5 configuration as development evaluation.
Export: `checkpoints/context-train-export-20261026`. The visual pretrained encoder
has seen the training corpus; this is explicitly not an out-of-fold export.
The limited source selection is not a random sample of the entire LRS3 corpus.

After training-ID checks and transcript deduplication against the development
references, seed42 selects1,000 clips for optimization and100 for internal loss
evaluation. Both arms use identical IDs, transcripts, presentation counts and
initial clean EOS adapter. Clean-only supplies each reference-phone example
twice; mixed supplies one clean example and one current-model rank-one example.
Both have2,000 training examples and200 internal-validation examples. Internal
validation input types differ between arms, so their loss values are not a
directly comparable accuracy metric.

One epoch,500 updates, batch4, lr1e-5,3% warmup and linear decay, BF16, LoRA
rank16/alpha32/dropout0.05, weight decay0.01, target-only CE with supervised EOS,
512-token complete-pair limit, no truncation. Save at250 and500. Each run is
capped at15 minutes of training with thermal guard and cooperative PAUSE/resume.
Preparation and final evaluation are separate overhead. No checkpoint is selected
using the external development result; evaluate the final step in both arms.

Run artifacts: `checkpoints/decoder-current-{clean,mixed}-20261027`.
Planned matched evaluation uses the cached context-model candidates on all1,140
development clips, followed by reference-phone checks on the same clips.
Generation, normalization and adapter initialization match the retained baseline.
Compare actual selected WER first, oracle@5 second; clean-phone improvement alone
does not justify visual-pipeline promotion. No fresh test set is used.

## Local coding worker and verification

The requested local DeepSeek worker was used for test coding. Its first attempt
timed out; another retry exposed an8K context configuration and was retried at32K.
The worker subsequently changed production filtering outside its assignment.
That edit was rejected and the verified transcript-deduplicating implementation
restored. The worker's final tests required supervising corrections for temporary
directory isolation and a false-positive overlap test. Twenty decoder tests pass
after review, and real-data checks confirm matching IDs and transcripts.
Worker output was not accepted on its claimed success alone. The idle coding
model was unloaded before training to release GPU memory.

## Results

Completed 2026-10-27. Both final trainer states confirm 500 updates and one
epoch. Training took 306.34 seconds for clean-only and 326.53 seconds for mixed.
All four evaluations contain 1,140 clips and 12,675 reference words. Model
revision, source hash, prompt, normalization and generation settings match the
retained visual baseline.

| Adapter | Visual WER (errors) | Oracle WER@5 (errors) | Reference-phone WER (errors) |
|---|---:|---:|---:|
| Retained clean EOS | 95.48% (12,102) | 92.53% (11,728) | 13.40% (1,699) |
| Clean-only continuation | 95.74% (12,135) | 91.40% (11,585) | 7.35% (931) |
| Mixed current-error continuation | 95.23% (12,070) | 90.12% (11,423) | 11.96% (1,516) |

Mixed training reduces selected visual WER by 0.25 percentage points versus
the retained baseline (32 fewer errors), and by 0.51 points versus the matched
clean-only control (65 fewer errors). Its oracle improves by 2.41 points versus
baseline, but that reference-aware choice is not a deployable ranking result.
Neither new adapter has an exact sentence among its five visual candidates.
Clean-only improves reference-phone reconstruction substantially while slightly
worsening selected visual WER. Mixed training sacrifices 4.62 points of clean
reconstruction WER relative to that control.

Visual/reference generation caps were 2/0 for clean and 7/2 for mixed. Reported
evaluation runtimes were 82.13/42.23 seconds and 104.77/55.07 seconds respectively;
these are evaluator timings, not total model-loading and thermal-wait wall time.
Metrics are in each run's `visual-eval/metrics.json` and
`reference-eval/metrics.json`. Twenty decoder tests pass after completion.

Decision: retain the existing baseline and keep mixed as an experimental
candidate. The selected-output gain is small, comes from one seed on repeatedly
inspected development data, and has no independent test confirmation. This
pilot supports testing noise adaptation further, not scaling training blindly.
The next proposed check is fixed-candidate ranking followed by a repeated-seed
comparison if authorized. No further training was launched.

New code and results are not part of the earlier eight commits already pushed
to origin/visual-phoneme.
