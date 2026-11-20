# Route-SFT action-type BCE

This is a single-objective addition to the original Route-SFT recipe. It does
not incorporate the separate experimental marker initialization, class weights,
Good KL, layer freezing, or constrained decoding changes. No improvement in
completion rate or image quality is claimed without matched rollout evaluation.

## Objective

For each example, let `r` be the existing reasoning-error label and `yA` indicate
that the **supervised target action** starts with `<answer>`. Set `yT = 1 - yA`.
Good retains its natural action; Bad uses its reviewed correction. Neither label
nor action-type weight is inferred from Good/Bad.

| Example | Route label | Tool label | Answer label |
| --- | ---: | ---: | ---: |
| Good, tool target | 0 | 1 | 0 |
| Good, answer target | 0 | 0 | 1 |
| Bad, corrected tool target | 1 | 1 | 0 |
| Bad, corrected answer target | 1 | 0 | 1 |

The original route predictor is at `target_start - 1`. The first action begins
at `a = target_start + r`, after the optional marker. Its predictor is `a - 1`.
The action prefix `s = input_ids[:a]` contains H/R and, for Bad, `<RETHINK>`; it
does **not** contain any action token or action body.

The pinned tokenizer encodes `<tool_call>` as `[151657]` and `<answer` as
`[27, 9217]`. The next answer token can merge `>` with a newline or `{"`.
We therefore score the distinguishing `<answer` prefix, not an invented
single-token `<answer>` and not the first ordinary `<` alone:

\[
P_T=p_\theta(151657\mid s),\qquad
P_A=p_\theta(27\mid s)\,p_\theta(9217\mid s,27).
\]

These probabilities come from the full autoregressive vocabulary. They are NOT
renormalized by `P_T + P_A`; they need not sum to one. Completion of the tags,
JSON, body and EOS remains under the original suffix CE. This BCE does not claim
to measure full-program validity.

\[
\mathcal L=\frac1B\sum_i\left[
\frac1{L_i}\sum_t\operatorname{CE}_{it}
+0.1\operatorname{BCE}(q_i,r_i)
+\frac{\lambda_{\rm type}}2\left(
\operatorname{BCE}(P_{T,i},y_{T,i})+
\operatorname{BCE}(P_{A,i},y_{A,i})\right)\right].
\]

Default `type_lambda=0.1`. Each example contributes one two-class mean, followed
by the existing global sample mean. There is no second division by batch size,
no action-body length normalization for type BCE, no positive-class upweighting,
and no requirement that the two target classes occur equally often. The 6:2
Good/Bad sampler and sealed sample order are unchanged. The marker remains in
the suffix CE as before, in addition to its independent route BCE.

## Stable probabilities and causal replay

For a token, compute its binary logit against all other vocabulary entries:
`r_v = z_v - logsumexp(z_other)`. For the two-token answer prefix, use

\[
\operatorname{logit}(P_A)=
-\operatorname{logsumexp}(-r_{<},-r_{answer\mid<},-r_{<}-r_{answer\mid<}).
\]

`BCEWithLogitsLoss` can then be used without clamping probabilities near zero
or one. Both candidate branches receive gradients; no parent or extra head is
introduced. Full-vocabulary type logits are calculated in FP32.

Each microbatch first performs the unchanged teacher-forced suffix forward,
then one additional differentiable forward of **only `s + [27]`** to obtain
`p(answer | s, <)`. This second pass is executed for BOTH target types on EVERY
rank when the objective is enabled. Skipping it on answer examples would make
the number of FSDP forward/backward collectives depend on each rank's labels.
No action body or future response is appended to the replay. Native positions,
history and tokenizer are unchanged; the replay is an auxiliary score, never a
tool action or an extra Actor observation.

This is not zero-cost: without a shared differentiable KV cache the additional
pass reprocesses the prefix. No speed or peak-memory claim is made for four
A100s before a real GPU preflight. The existing gradient checkpointing and
FSDP policies are retained.

## Reproduction and diagnostics

Use the README's normal `check-route` and `route-sft` / `all` commands with a
**new WORK directory**, starting from the same GenEvolve 2B SFT-550 parent.
`configs/route_sft.json` records both the coefficient and objective version in
the checkpoint contract. Old Route-SFT runs/exports must not be silently resumed
or reused as this new objective. Their files are not deleted.

To run the single-variable baseline, set only `type_lambda` to `0.0` before
starting another new WORK. This disables the candidate forward and restores the
original suffix CE + route BCE computation. Data, learning rate, epochs, marker
initialization, language-layer training and inference settings remain unchanged.
Stage two still uses native GenEvolve GRPO + SDL with Actor-routed Teacher-only
hints; this change does not modify RL or its decoding.

Per-step logs retain the original metrics and add `tool_bce`, `answer_bce`,
`type_bce` (their mean), `type_weighted`, `route_weighted`, `p_tool`, `p_answer`,
and the target-class counts/mean probabilities. These are teacher-forced train
metrics, not free-generation accuracy. Check the weighted terms and gradients;
batch averaging alone does not ensure type BCE is small compared with CE.

Tests cover all four label combinations, native merged-tag variants, positions,
future-body isolation, extreme logits, gradients through both branches, the
zero-weight baseline, and loss/batch averaging. Linux CI additionally checks
four-rank CPU/FSDP mixed-label updates against a nonsharded global-batch reference
and saves a verified training checkpoint. CPU tests do not establish GPU memory
fit, API reliability, completion quality or benchmark improvement.
