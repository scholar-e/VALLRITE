# Repository instructions for AI agents

## GPU execution

GPU work is supported on the host NVIDIA RTX 5090. The restricted command
sandbox may hide the GPU, so run CUDA training, inference, and GPU diagnostics
with host/elevated execution rather than concluding that CUDA is unavailable.

Use the checked-in launcher for GPU Python commands:

```bash
tools/with-vpa-gpu .venv-vpa-gpu/bin/python -m VisualPhoneme.train ...
tools/with-vpa-gpu .venv-vpa-gpu/bin/python -m AudioPhonemeLabeler ...
```

The launcher adds Ollama's CUDA 12 runtime and the virtual environment's cuDNN
directory to `LD_LIBRARY_PATH`. This is required by CTranslate2/faster-whisper,
which otherwise reports a missing `libcublas.so.12`. Do not install replacement
CUDA packages until the launcher has been tried. Confirm availability with:

```bash
tools/with-vpa-gpu .venv-vpa-gpu/bin/python -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Keep training runs bounded with the program's `--max-minutes` option when the
user provides a time budget. GPU visibility and library loading are separate:
the launcher fixes libraries, while host/elevated execution exposes the device.
