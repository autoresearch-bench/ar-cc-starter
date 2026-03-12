# Observations

## Baseline Architecture
- GPT-style transformer: 6 layers, 512 dim, 8 heads
- Vocab size: 8192 (BPE), Context: 2048 tokens
- Parameters: ~19.5M (estimated)
- Optimizer: AdamW, LR=3e-4, 100 warmup steps, cosine decay
- Batch size: 64
- Training: 5 minutes on H100, bfloat16 autocast
- No gradient clipping, no weight decay specified (AdamW default=0.01)

## Key Constraints
- TIME_BUDGET = 300 seconds (5 min wall clock)
- H100 GPU (~80GB VRAM)
- EVAL_TOKENS = ~21M tokens
- Vocab size fixed at 8192
- MAX_SEQ_LEN = 2048
- Must keep model interface: forward(x, y, reduction='none')

## Experiment Results
| Experiment | val_bpb | Notes |
|-----------|---------|-------|
| Baseline  | TBD     | 6L/512d/8h, BS=64, LR=3e-4 |

## Key Learnings
(to be filled as experiments complete)
