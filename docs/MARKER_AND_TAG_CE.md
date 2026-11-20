# Marker initialization + class-balanced native tag CE

This recipe keeps only two mechanisms from the experimental repair: a frozen-parent
marker-row initialization, and splitting opening-tag CE from the remaining suffix CE.
It **replaces** the earlier tool/answer type BCE; it does not stack both objectives.
The previous type-BCE implementation remains available in Git history at `50f6809`.

## Unchanged contract

- Same 2B GenEvolve SFT-550 parent, reviewed-v3 public data, masks and ordered batches.
- 2,376 examples: 1,782 Good / 594 Bad; 6:2 per global batch of eight.
- One epoch / 297 optimizer updates, four ranks, microbatch one, accumulation two.
- LR 2e-6, AdamW (0.9, 0.95), decay .01, warmup .1, cosine, clip 1, seed zero.
- Maximum 12,288 tokens, no truncation; visual tower frozen, all text layers trainable.
- No reasoning CE, reference/Good KL, additional text-layer freezing or LoRA.
- Original full-vocabulary route BCE, coefficient .1, **no positive-class upweighting
  during normal Route-SFT** and no action-candidate conditional route BCE.
- Native `<think>...</think>` followed by optional `<RETHINK>` and `<tool_call>` or
  `<answer>`; no action wrapper, marker logits processor, history stripping or budget change.
- Stage-2 GRPO/SDL, Actor-routed Teacher-only hints, evaluation and sampling unchanged.

## 1. Calibrate only the tied marker row

Before ordinary Route-SFT, freeze the unmodified SFT-550 parent. For each training
row, pass **only** `input_ids[:target_start]`. Extract its final hidden state
\(h_i\) and the original full-vocabulary competitor
\(c_i=\operatorname{logsumexp}_{v\ne\tau}z_{iv}\), where \(\tau=151669\).
Neither the target action nor future feedback is in this prefix. These prefixes
contain no marker, so changing its input-embedding row cannot change these states.

The public implementation fixes the initialization to equal action-kind mass:

\[
w_0=\tfrac12 W_{151657}+\tfrac14 W_{27}+\tfrac14 W_{9217}.
\]

These are `<tool_call>`, `<`, and `answer`. On a deterministic task-grouped internal
fit split, optimize one vector, with the parent frozen:

\[
\widehat w=\arg\min_w\frac1{|I|}\sum_{i\in I}
\operatorname{BCEWithLogits}(h_i^\top w-c_i,r_i;\mathrm{pos\_weight}=0.5)
+\frac{0.01}{2}\lVert w-w_0\rVert_2^2.
\]

Use `SHA256("frozen_parent_tied_row_v1:" + source_task_id) mod 5`; remainder zero
is diagnostic-only. Missing task IDs fall back to the existing same-state hash,
then row ID. All examples still participate in subsequent Route-SFT. The internal
diagnostic is not an external validation set and does not select a candidate,
threshold or iteration. There is no hard-coded claim about its AUC or sample counts.

**Implementation details are explicitly fixed here, not recovered from unpublished
experiment scripts:** CPU float64 full-batch L-BFGS, strong-Wolfe line search,
LR 1, at most 200 iterations, gradient tolerance 1e-9 and change tolerance 1e-12.
The action-vector mixture, exact hash rule, optimizer and ridge-around-initial-vector
convention are this repository's reproducible choices. The two mechanisms and
reported coefficients are retained; this is not a byte-identical reconstruction
of an unavailable experimental seed export. No extra hyperparameter sweep is run.

Copy the fitted vector into **only the tied embedding/LM-head row** before FSDP
wrapping. Untied/bias-bearing heads are rejected; other weights remain exactly
the parent weights. The parent directory is never modified and no second full
checkpoint is created. Marker registration accepts both first registration and
an already-correct special token, with the ID checked in either case.

The usual `route-sft` and `all` entrypoints perform this automatically. Four GPUs
extract independent frozen-parent feature shards, then CPU fits the small vector;
the frozen replicas are released before the normal training model is loaded.
This adds one prefix-only forward pass over the training pack, **not** extra
language-model training epochs or API calls. The GPU time is not yet measured.

Caches under `WORK/route_sft/marker_initialization` include per-file SHA-256 receipts,
parent/data/config identities, `seed.pt`, and `seed_provenance.json`. Unchanged
completed feature shards can be reused after an interruption. Partial/corrupt or
identity-mismatched receipts fail explicitly rather than being silently trusted.
On training resume, the original seed receipt is required and the saved Actor and
optimizer are restored; the seed is **not applied again**. Preserve this small
directory together with training checkpoints for continuation/audit.

## 2. Separate the action opening tag from the remaining suffix

Let \(s_i\) be `target_start`, \(a_i=s_i+r_i\), and let \(b_i\) be the exclusive
end of the actual token span covering the complete native opening tag. Good and
Bad use the same objective; the optional marker merely shifts the action start.
Use the **target** action kind, not the original action kind or Good/Bad label.

Decode the sealed IDs to locate the tag; never retokenize or insert a separator.
`<tool_call>` is one token. `<answer>` usually takes three tokens; its last token
can include `>\n` or `>{"`. Such a crossing BPE token is indivisible and belongs
to the tag once. All remaining supervised tokens, including Bad's marker,
the closing tag and EOS, belong to the content part. No token is dropped or
counted in both CE parts; history/reasoning stay masked as before.

\[
\mathcal L=\frac1B\sum_i\left[
0.15\frac1{|C_i|}\sum_{t\in C_i}\operatorname{CE}_{it}
+0.5w_{k_i}\frac1{|T_i|}\sum_{t\in T_i}\operatorname{CE}_{it}
+0.1\operatorname{BCE}(q_i,r_i)\right],
\quad
w_k=\frac{N}{2N_k},\quad
q_i=\operatorname{softmax}(z_{s_i-1})_\tau.
\]

Here \(T_i=[a_i,b_i)\) and \(C_i=[s_i,L_i)\setminus T_i\). Counts and weights
are computed over the entire sealed training pack, **not each minibatch**:
621 answers -> 1.91304348, 1,755 tools -> 0.67692308. The two kinds receive equal
total tag-weight mass over the full dataset; they need not have equal losses,
per-token gradients, or counts in any one batch. There is one global sample mean,
not an additional division by batch size. The tag coefficient is independent of
action-body length. This is class-balanced CE, not tool/answer BCE or a two-way
normalization of the full vocabulary.

Both terms use the original teacher-forced forward. No extra candidate-prefix
forward is needed. `suffix_ce` is retained as the old unweighted suffix-mean
diagnostic; it is **not** added again to the total. Logs expose `content_ce`,
`action_tag_ce`, `content_weighted`, `action_tag_weighted`, `weighted_ce`,
`route_bce`, `route_weighted`, per-kind counts/tag CE, and Good/Bad marker means.
These are training-prefix measurements, not free-generation scores.

## Start and verification

Use a **new WORK** and the original matched SFT-550 parent with the README commands.
Old type-BCE or multi-repair checkpoints cannot resume or enter stage two as this
recipe: the export/continuation contract checks the entire configuration.

CPU tests cover frozen-prefix extraction, only-one-tied-row mutation, deterministic
fit and diagnostic isolation, receipt validation and resume caching, full opening
tags including merged BPE boundaries, all four route/action combinations, exact
loss/gradients, length-independent tag weights, EOS/marker coverage and global
batch averaging. Linux CI also checks four-rank CPU/FSDP updates and continuation.
These tests do not establish four-A100 memory/throughput, free-generation RETHINK
behavior, completion rate or benchmark gains. Those require a matched GPU run.
