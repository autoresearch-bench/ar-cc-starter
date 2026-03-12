# Research Plan

## Current Phase: Baseline + Initial Explorations

### Priority 1: Establish Baseline
- Run the default train.py to get baseline val_bpb

### Priority 2: Quick Architecture Wins (parallel)
These are low-risk, likely-to-improve changes:

1. **Increase model size** — The H100 has 80GB VRAM and the baseline uses minimal memory. Scale up n_embd, n_layer, and n_head to use more GPU capacity.
2. **Gradient clipping** — Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` to stabilize training.
3. **Weight decay tuning** — AdamW default is 0.01. Try 0.1 which is common for transformers.
4. **Larger batch size** — With H100 memory, we can afford much larger batches. Try 128 or 256.
5. **Learning rate tuning** — Try higher LR (e.g. 1e-3) which may work with larger batch.
6. **torch.compile** — Use `model = torch.compile(model)` for faster training throughput, enabling more steps in 5 min.

### Priority 3: Architecture Improvements
- RMSNorm instead of LayerNorm
- SwiGLU activation instead of GELU
- Rotary position embeddings (RoPE) instead of learned
- Tie embedding weights (tok_emb and lm_head)

### Priority 4: Training Optimizations
- Gradient accumulation if batch size is limited
- Mixed precision with loss scaling
- Learning rate schedule tuning (different warmup, min LR)

### Priority 5: Advanced
- MuP parameterization
- z-loss regularization
- Different optimizer (e.g., SOAP, Lion)

## Budget Strategy
- 2.0 GPU-hours budget = ~24 five-minute experiments
- Phase 1: Run 4-6 experiments in parallel exploring major axes
- Phase 2: Combine best findings
- Phase 3: Fine-tune hyperparameters
