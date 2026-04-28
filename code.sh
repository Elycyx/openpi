uv run scripts/serve_policy.py \
    --spec.enable-speculative \
    --spec.spec-batch-size 8 \
    --spec.spec-diffusion-num-steps 10 \
    --spec.spec-delta-thresholds 0.9 1.4 1.3 1.2 1.2 1.3 \
    --spec.action-env-dim 7 \
    --spec.spec-debug \
    --spec.spec-log-path spec.log \
    policy:checkpoint \
    --policy.config=pi05_piper_3in1 \
    --policy.dir=checkpoints/pi05_piper_3in1/27000



uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper_3in1 --policy.dir=checkpoints/pi05_piper_3in1/27000
