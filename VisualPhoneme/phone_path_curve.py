"""Oracle phone error rate vs number of beam paths kept, head C, test-select.

Usage: python -m VisualPhoneme.phone_path_curve BEAM_WIDTH (64 is what the exports use)."""
import json, sys, time
from pathlib import Path
import torch
from VisualPhoneme.data import ctc_prefix_beam_search
from VisualPhoneme.evaluate_autoavsr_ctc import edit_counts
from VisualPhoneme.train_autoavsr_phone_head import Split, build_head

NS = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 64, 100, 128, 200, 256]
width = int(sys.argv[1])
ckpt = torch.load("checkpoints/autoavsr-phone-C-masked-20261106/best.pt", map_location="cpu", weights_only=True)
split = Split(Path("checkpoints/autoavsr-features-20261102/test"), torch.device("cpu"))
head = build_head(ckpt["config"], split.features.shape[1]); head.load_state_dict(ckpt["head"]); head.eval()
indices, ns = list(range(660)), [n for n in NS if n <= width]
oracle, phones, kept, t0 = {n: 0 for n in ns}, 0, 0, time.time()
with torch.inference_mode():
    for start in range(0, 660, 64):
        chunk = indices[start:start + 64]
        feats, lengths, _, _ = split.batch(chunk, head.upsample)
        logp = head(feats, lengths // head.upsample)
        for slot, i in enumerate(chunk):
            ref = split.rows[i]["phone_ids"]
            cands = ctc_prefix_beam_search(logp[slot, :int(lengths[slot])].double(), beam_width=width,
                                           top_n=width, token_top_k=12)
            errs = [sum(edit_counts(ref, s)) for s, _ in cands]
            kept += len(cands); phones += len(ref)
            for n in ns:
                oracle[n] += min(errs[:n])
print(json.dumps({"beam_width": width, "mean_paths": kept / 660, "seconds": round(time.time() - t0),
                  "oracle_per": {n: round(100 * oracle[n] / phones, 2) for n in ns}}))
