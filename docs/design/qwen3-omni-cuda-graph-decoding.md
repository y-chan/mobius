# Qwen3-Omni thinker: CUDA-graph decoding

What an exported Qwen3-Omni thinker needs from the build command, the ONNX
Runtime session and the driving loop in order to decode at speed. Reading the
model module alone does not reveal most of it: the graph is correct without any
of this, just several times slower, and two of the settings corrupt the output
rather than failing.

Numbers below are the 30B checkpoint with int8 routed experts on an RTX PRO
6000 Blackwell Max-Q (ORT 1.29, CUDA 12.9), batch 1, a 138-token prompt and
greedy decoding. The session and loop requirements hold on any CUDA device;
**the per-kernel timings do not transfer between architectures**, and one of
the graph-level wins below is specific to Blackwell. Measure on the device you
serve from before assuming a figure carries over.

## Build

```bash
mobius build --config <olive-int8-checkpoint> \
    --dtype f16 --ep cuda --features static-cache --max-seq-len 4096 \
    --output <out>
```

`--ep cuda` is not the default and is not optional for performance.
`--execution-provider` defaults to `default` (portable ONNX, no vendor
fusions), and the EP decides two things that matter here:

- `supports_transposed_matmul` makes the attention projections emit
  `com.microsoft::FusedMatMul` with `transB=1` against the checkpoint's
  `[out, in]` weight. Without it they are a `MatMul` against a pre-transposed
  `[in, out]` initializer, which on Blackwell at batch 1 is the slow layout
  for cuBLAS: the 48-layer decode step spends 3.67 ms there instead of
  1.58 ms (118 vs 154 tokens/s overall).

  **This one is architecture-dependent.** On an A100 80GB PCIe (SM80) the two
  layouts are a wash, because the plain layout is already fast there: per
  projection, `[1,2048] x [2048,5120]` goes 37.4 → 35.1 µs and
  `[1,4096] x [4096,2048]` goes 38.1 → 41.0 µs, so the transposed form is a
  net loss of about 0.6 µs per layer. It costs nothing to leave on, but do not
  expect the Blackwell figure anywhere else.
- `supports_fused_moe` keeps one MoE node per layer. The loop-over-experts
  fallback materialises 18k expert tensors for this checkpoint and spends
  about twenty minutes in session initialisation.

`--features lm-head-int8` does the same for the LM head: 6.01 → 5.79 ms per
decode step (167 → 173 tokens/s), the head itself 0.387 → 0.186 ms. One GEMV
per step against the projections' 48, so a smaller prize -- but a cheaper one
than its position suggests. Over 35 teacher-forced positions, quantizing the
head moves the logits a seventh as far as quantizing the projections does
(mean |Δ| 0.019 against 0.134, worst 0.15 against 3.1), because the head
applies once where a projection's error compounds through 48 layers. It
refuses a checkpoint with tied word embeddings, where quantizing the head
would untie it and store the vocabulary twice.

`--features attn-int8` quantizes the two attention projections at export
time and emits `com.microsoft::MatMulNBits` (8-bit, symmetric, one scale per
`--weight-block-size` input features, no zero-points) in place of the float
matmul. This is weight-only -- W8A16: the op dequantizes the weight and
computes in the activation's dtype, and nothing quantizes the activations.
W8A8 is not on offer because ORT's CUDA EP has no fast kernel for it
(`DynamicQuantizeLinear` is unimplemented there and `MatMulInteger` is slow).
Routed experts and the LM head are untouched. It requires `--dtype
f16`, since ORT's MatMulNBits CUDA kernel reads fp16 scales, and the build
refuses bf16 rather than emitting a graph that cannot run.

| | A100, GQA cache | Blackwell, static cache |
|---|---|---|
| decode step | 6.81 → 6.43 ms | 6.51 → 6.01 ms |
| tokens/s | 147 → 155 | 154 → 167 |
| prefill | +4% | +1% |

This is the one optimisation here that has held up on both architectures.
Blackwell gets more of it: the decoder's `Run` goes 5.43 → 4.86 ms, and the
greedy tokens still match the HF reference on all ten runs of the test clip.

`--features gqa-cache`, alongside `static-cache`, hands the cache to
`com.microsoft::GroupQueryAttention` instead of `TensorScatter` + `Attention`.
It changes the decoder's I/O and is worth it only on some devices and prompt
lengths; the last section has the numbers.

`--features gqa-fused-rope`, with `gqa-cache`, additionally gives that node
the Q/K norm weights and the rotary tables so neither is emitted per layer.
It takes the decoder's `position_ids` input away, because the op derives the
rotary position from the KV length instead, and it is only correct while
position_ids carry the same value on all three mRoPE axes. Measured 6.51 ->
6.16 ms per decode step on an RTX PRO 6000, tokens unchanged.

`--features static-cache` is a prerequisite for graph capture, not an
optimisation on its own: it replaces growing `past_key_values` and
`attention_mask` with pre-allocated `key_cache.{i}` / `value_cache.{i}`
buffers plus `write_indices` and `nonpad_kv_seqlen`, so every single-token
decode step has one shape.

`--max-seq-len` does not affect decode speed. Attention over the cache is
bound by the *valid* KV length, not by the buffer: 1024 and 4096 slots measure
the same end to end, and 4096 against 8192 slots at a fixed context differ by
0.002 ms over 48 layers. Size it for the longest sequence you serve. The
actual context does cost something — on an A100, 170 against 422 past tokens
is 1.049 against 1.224 ms over 48 layers — so prompt length, not buffer
length, is what shows up.

## Session options

| Option | Why |
|---|---|
| `enable_cuda_graph=1` (CUDA EP provider option) | 86 → 118 tokens/s. Capture the decode step only; see below. |
| `session.use_device_allocator_for_initializers=1` | Allocates weights outside the session arena. Without it the arena's rounding holds 18 GiB of unused device memory (58.9 → 76.1 GiB for the fp16 export). |

Two things to leave alone:

- **Do not set `arena_extend_strategy=kSameAsRequested`.** It reduces reserved
  memory and is incompatible with graph capture: replay assumes the addresses
  recorded at capture time, so the output is wrong and varies between runs.
  There is no error — only bad tokens, intermittently enough to be mistaken
  for a measurement artefact.
- **Keep the CPU EP in the provider list.** A few shape-computing nodes
  (`Shape` / `Slice` / `Concat` over `position_ids`) prefer CPU. They are
  constant across decode steps, so a captured graph is not wrong because of
  them. `session.disable_cpu_ep_fallback` requires removing the CPU EP from
  the list and then fails on nodes with no CUDA kernel.

## Driving loop

Capture the decode step and nothing else. Prefill keeps varying shapes, so run
it with the run option `gpu_graph_id=-1`; it writes into the same device cache
buffers the captured decode graph reads.

Every decode input and output must live at a fixed device address:

- Bind through `IoBinding` with `OrtValue`s allocated once. Update them in
  place (`update_inplace`) rather than rebinding.
- Bind each KV cache buffer as both the input `key_cache.{i}` and the output
  `updated_key_cache.{i}`; `TensorScatter` then updates it in place.
- Pre-allocate the logits buffer with `ortvalue_from_shape_and_type` and bind
  it. With capture enabled the session returns non-tensor placeholders for
  outputs it did not allocate itself, so reading `get_outputs()` fails.
- If the loop does create a binding per step, keep the previous step's binding
  alive. The `OrtValue`s from `get_outputs()` are owned by their binding, and
  releasing it raises `Integer overflow` on the next step.

**Discard two generations after capture.** The capture itself (the first
generation) returns correct tokens, but the first *replay* after it does not;
everything from the third generation on is correct. This reproduces across
exports and is independent of the model, so a server should run two dummy
generations between session creation and the first real request.

One optimisation to skip: writing the embedding straight into the captured
graph's input buffer produces wrong tokens, and `synchronize_outputs()` does
not fix it. Route the decode embedding through the host — it is one
`1 x hidden` copy per step, 0.06 ms.

## Where a decode step goes

6.54 ms per step as driven by `omni_check.py`:

| | ms |
|---|---|
| Decoder `Run` (4.80 ms of kernels, 0.62 ms of graph launch and sync) | 5.43 |
| Host-side `isfinite` + `argmax` over the 151,936-wide logits row | 0.78 |
| Logits fetch (304 KB device-to-host) | 0.12 |
| Embedding session `Run` | 0.09 |
| Input transfers | 0.06 |

Host-side I/O totals 0.27 ms, so neither the embedding round trip nor the
logits transfer is worth restructuring. The `isfinite` check is a validation
guard, not something a server does; without it the step is 5.76 ms
(174 tokens/s), which is where this export lands against vLLM's int8 path
(5.71 ms, 175 tokens/s, 34.0 GiB of weights against 33.6 GiB here).

Within the 4.80 ms of kernels, the attention projections take 1.58 ms and the
routed experts 1.68 ms, both at roughly 1.4 TB/s of effective bandwidth. The
remainder is the LM head (0.39), attention itself (0.44), the two norms (0.40)
and RoPE, the cache write, the QKV split and the router gate (0.47).

## Measuring this

`SessionOptions.enable_profiling` is not usable for a per-operator breakdown
here. Its instrumentation cost scales with the node count — the step goes from
11.7 ms to 26.8 ms under profiling — so the result ranks operators by how many
nodes they have. It attributes 29% to `MatMul` and 11% to `Reshape`, which
launches no kernel at all.

Use Nsight Systems with `--cuda-graph-trace=node`; without it the whole
captured graph collapses into one range. Two cautions:

- Restrict the summary to the generation window. The cutlass tactic sweep for
  the MoE kernels runs during the *first* generation, so totals over the whole
  capture put `MoeFCGemm` at 43% of decode time. If
  `populateRandomBufferKernel` appears in the output, the window still
  includes the sweep.
- The marginal cost of one kernel inside a captured graph is about 0.1 µs, not
  the 1.6 µs that dividing total idle time by kernel count suggests. Node
  count is not a lever once the step is captured — packing Q/K/V into one
  projection removes 96 `MatMul` nodes per step and changes nothing
  measurable, and replacing `TensorScatter` + `Attention` with
  `com.microsoft::GroupQueryAttention` is slightly *slower* in a graph
  (0.637 ms against 0.608 ms over 48 layers) even though it wins by a wide
  margin outside one. **That verdict holds only for Blackwell and only for
  short contexts** — see below.

## GroupQueryAttention and context length (SM80)

The static-cache path costs what the *valid* KV length costs, and how steeply
depends on the architecture. Measured over 48 layers with an 8192-slot buffer,
`TensorScatter` + opset-24 `Attention` against `com.microsoft::GroupQueryAttention`
on the same graph shapes:

| past tokens | Blackwell static | Blackwell GQA | A100 static | A100 GQA |
|---|---|---|---|---|
| 170 | 0.753 ms | 0.727 ms | 1.351 ms | 1.369 ms |
| 422 | 0.799 | 0.790 | 1.530 | 1.420 |
| 1024 | 0.801 | 0.810 | 1.689 | 1.435 |
| 2048 | 0.916 | 0.901 | 2.044 | 1.463 |
| 4096 | 1.011 | 1.007 | 3.039 | 1.596 |

On Blackwell the two are indistinguishable at every length, and both scale
gently (0.07 µs per token of context). On an A100 the static path scales six
times more steeply (0.41 µs per token) while GQA stays nearly flat
(0.05 µs per token), so by 4096 tokens of context GQA is 1.4 ms ahead over the
48 layers.

This is not an ONNX Runtime regression: 1.29 shows the same shape, with the
static path steeper still (0.58 µs per token) and GQA at 0.047. ORT 1.30
improves the static path on SM80 without changing the asymptotics.

So the attention op to emit is a function of the device and of how long the
prompts are, which is why it is a build feature rather than a default:

```bash
mobius build ... --features static-cache,gqa-cache
```

Short prompts, or Blackwell, and the static path as exported is already the
right choice. Long prompts on SM80 and `gqa-cache` pays for itself.

It changes the decoder's I/O, which is why it is opt-in. The buffers become
4-D — `[B, kv_heads, max_seq_len, head_dim]`, the layout the op requires,
against `[B, max_seq_len, kv_hidden]` for the TensorScatter path — and
`write_indices` disappears, because GQA derives the write position from the KV
length itself. `nonpad_kv_seqlen` stays, and the `seqlens_k` / `total_seq_len`
pair GQA takes is produced inside the graph, so no other input changes. A
driving loop therefore needs new cache buffers and one fewer feed; everything
else about it is unchanged.

One detail of that is load-bearing. `seqlens_k` is computed from
`nonpad_kv_seqlen`, but `total_sequence_length` is emitted as a **constant**,
the buffer's capacity. ORT wants that scalar on the host, so deriving it from
a device tensor puts a device-to-host copy in the decode step and graph
capture fails outright. A constant needs no copy. It is sound because the op
bounds the KV it reads by `seqlens_k`: against the true length it measures the
same (0.744 against 0.720 ms over 48 layers at 170 past tokens, identical at
4096) and produces the same logits as the TensorScatter path.

The feature is wired for speech-language exports (Qwen3-ASR,
Qwen3-forced-aligner, Qwen3-Omni). The causal-LM and Gemma4 static-cache paths
still emit `TensorScatter` + `Attention`; extending them is mechanical, and
Gemma4 additionally needs its bias layers to stay on the Attention op, which
is the one thing GQA cannot express.

## Quantizing the attention projections

`--features attn-int8` is the one place where mobius adds quantization rather
than carrying the checkpoint's own. The weights are quantized in
`preprocess_weights`, from whatever dtype the checkpoint stores. For this
model that costs nothing either way — bf16 has 8 mantissa bits against fp16's
10, so the fp16 export holds the same values — but it is where a checkpoint
whose dtype a narrowed export *would* round belongs, and it keeps the
quantization out of a post-export rewrite.

The emission is `QuantizedLinear`, the component the GPTQ and AWQ paths
already use, injected through `linear_class`. At 8 bits nothing is bit-packed,
so its `[N, n_blocks, blob_size]` payload is exactly `[out, in/block, block]`.
Zero-points are left off so the op uses its implicit 128. `accuracy_level` --
the attribute that makes ORT's MLAS kernel quantize activations to int8, i.e.
the one that would turn this into W8A8 on CPU -- is omitted for anything other
than 4 bits, so the 8-bit path stays weight-only on every EP.

The scale is rounded to the model dtype *before* the weights are divided by
it, so the stored integers fit the scale the kernel will actually read.

`--weight-block-size` trades weight error against almost nothing. The payload is
one byte per weight whatever the block, so only the scales change size, and at
block 128 they are 1.6% of a projection's bytes. Measured over 6 layers of the
thinker, and end to end on Blackwell:

| block | mean error | worst | decoder `Run` |
|---|---|---|---|
| 64 | 0.66% | 0.81% | 4.884 ms |
| 128 (default) | 0.73% | 0.93% | 4.862 ms |
| 256 | 0.81% | 1.05% | not measured |

So the kernel does not care about the block size — 64 costs the 0.45% its
extra scale traffic predicts, nothing more — and the default is 128 only
because that is the size the end-to-end quality study below used. A study at
64 would be the thing that justifies moving it.

Both features together take the step to 5.79 ms, 4.97 ms of it the model
rather than the test harness, against vLLM's int8 path at 5.71 ms on the same
GPU. Neither changes the greedy tokens on the test clip: all ten runs still
match the HF reference, and the argmax agrees at all 35 forced positions.

What this does to the output beyond that has only been checked on one clip of
300 steps:
against the float export, the argmax agrees at 99.0% of positions and the
median KL is 4.9e-07, while agreement with vLLM's tokens is 98.3% against the
float export's 98.0%. One position differed on a digit (`1` → `2`) where the
float export was right. The worst-case KL of 0.16 is larger than a fake-quant
study of the same scheme suggested (0.024), so **several clips are still
needed** before treating this as settled.

## INT4 compressed-tensors checkpoints on a 24 GB card (RTX 3090)

llm-compressor's W4A16 output -- compressed-tensors `pack-quantized`,
symmetric INT4 in power-of-two groups, from GPTQ, AWQ or RTN -- exports with
every quantized tensor bound as stored. The packed int32 words of the
attention projections and of each expert already are the 4-bit
`MatMulNBits` / `QMoE` layout (low nibble first, offset by 8, which is the
ops' implicit zero point), so the export reinterprets them rather than
requantizing. The router, norms, embeddings, LM head and audio encoder stay
float, as the checkpoint's `ignore` list leaves them. At group 32 the thinker
is 20.5 GB, small enough for a 24 GB card with room for the static cache.

```bash
mobius build --config <compressed-tensors-w4a16-checkpoint> \
    --dtype f16 --ep cuda --features static-cache,gqa-cache,gqa-fused-rope \
    --max-seq-len 4096 --output <out>
```

Two things about the build matter on a small host:

- **Weights stream.** `build_pack_quantized_streaming_plan` reads one expert
  bank's sources and writes one destination at a time: 2.5 GB peak RSS and
  102 s for the 30B AWQ build
  (`cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit`, group 32) on a 16-core,
  15 GB machine, where the eager loader would hold the 20 GB checkpoint. A
  bank reads its 256 per-expert sources with one `safe_open` per shard;
  reopening the shard for every tensor produced the same bytes in 964 s.
- **QMoE's fc1 rows are written gate/up-interleaved**, so the graph emits no
  reorder. With the reorder in the graph, ORT folds it at session load and
  materialises every expert bank a second time on the host -- about 11 GB
  here, which a 15 GB host does not have to spare.

On the device the three sessions hold 19.1 GiB after creation and 22.3 GiB
after the longest prefill below (the arena keeps its peak). Session creation
takes 29 s.

Measured on an RTX 3090 (SM86, ORT 1.29), batch 1, greedy decoding, with the
driving loop above: the decode embedding routed through the host, two
generations discarded after capture. The prompt is a fixed system and user
turn around the audio, 162 text tokens plus 13 audio tokens per second.
Prefill runs from the waveform to the first token: mel features on the CPU,
audio encoder, embedding, decoder. For these runs the decoder's hidden states
were sliced to the last position before the LM head, a graph edit outside
mobius: the speech-language task returns logits for every prompt position,
0.5 GB of fp16 at 1.7k tokens. vLLM 0.33 serves the same checkpoint for
comparison (bf16 activations, Marlin kernels, `LLM.generate`).

| audio | prompt tokens | prefill, ORT | prefill, vLLM | decode, ORT | decode, vLLM |
|---|---|---|---|---|---|
| 15 s | 357 | 83 ms | 98 ms | 176 tok/s | 175 tok/s |
| 30 s | 552 | 133 ms | 148 ms | 176 tok/s | 176 tok/s |
| 45 s | 747 | 167 ms | 194 ms | 175 tok/s | 174 tok/s |
| 60 s | 942 | 212 ms | 233 ms | 175 tok/s | 173 tok/s |
| 120 s | 1722 | 392 ms | 419 ms | 173 tok/s | 171 tok/s |

At 45 s of audio the prefill is 11 ms of mel features, 29 ms of audio
encoder, 1 ms of embedding and 123 ms of decoder.

The table is the export with three Q/K/V projections. Joining them into one
`MatMulNBits` (now the default for quantized attention) takes decode to 181
tokens/s on ORT 1.29 with the prefill unchanged. Per layer the INT4 Q/K/V
GEMVs go from 17.0 to 10.5 us: the two `N = 512` calls ran at 150-220 GB/s
against 510-630 GB/s for the large ones, so a fixed per-call cost, not the
bytes, was what they spent.

The decode embedding's host round trip -- the embedding session's `Run` plus
the 4 KB copy into the captured graph's input -- costs 0.13 ms per step on
this host, 2% of a step, against 0.06 ms on Blackwell. Binding the embedding
output straight into the decoder's input buffer instead (same or separate
CUDA streams, with or without the per-`Run` synchronization) produced the same
greedy tokens on three clips here and recovered only 1%: the cost is the
second session `Run`, not the copy. A decoder that looks the token up itself
inside the captured graph decoded 7% faster in a hand-edited export.

The greedy transcripts agree with vLLM's in content but not always in
formatting. With timestamps and speaker labels removed and numerals
normalised, 95-100% of characters match over five clips of 15-120 s; the
formatting flips between `MM:SS.mmm` and `HH:MM:SS.mmm` timestamps and between
Arabic and kanji numerals. vLLM flips at the same positions between its own
bf16 and fp16 activations (also 95-100%), so this checkpoint is sensitive
there rather than the export wrong. The exact check is the synthetic parity
test against dequantized HF weights, which separates a gate/up row swap
(4.6e-3) from the bound export (under 1e-5).

## Block-paged KV cache (`paged-gqa`, RTX 3090)

A static cache reserves `max_seq_len` per request whether or not the request
uses it, so on a 24 GB card next to the INT4 thinker (18,866 MiB with the audio
encoder and the embedding) it fits 8 fp16 slots of 4096 tokens, or 16 with an
8-bit cache. `--features static-cache,gqa-cache,gqa-fused-rope,paged-gqa`
exports the decoder on `com.microsoft::PagedAttention` instead: one pool of
fixed-size blocks per layer, shared by every request, each holding only the
blocks its context fills.

The driving loop owns the pool and a free list, and runs the scheduled
requests as one packed sequence:

- **Prefill** of one request: `inputs_embeds [1, L, hidden]`,
  `cumulative_sequence_length [0, L]`, `past_seqlens [0]`, its row of
  `block_table` (unused entries -1), `attention_metadata [L, L]` and
  `logits_indices [L - 1]`.
- **Decode** of `n` requests, one token each: `inputs_embeds [1, n, hidden]`,
  `cumulative_sequence_length 0..n`, each request's `past_seqlens` and
  block-table row, `logits_indices 0..n-1`, and `attention_metadata
  [1, capacity]`. The query bound of 1 is what lets the step take ONNX
  Runtime's split decode kernel; the KV bound is replay-wide, so one CUDA
  graph per request count (`gpu_graph_id = n`) stays valid as the contexts
  grow. Reserve a new block before a step whose token starts one.
- The block caches are bound as input and output to the same buffers (the op
  rejects anything else).

On ORT's default backends a PagedAttention decode step is slow on this card:
XQA does not fit its shared memory, so a step falls to the portable paged
decode kernel or to a gathered copy of the context, 3-8x the attention time
of GQA's split decode. The numbers below use the opt-in split decode path for
PagedAttention (`ORT_PAGED_ATTENTION_SPLIT_DECODE=1`, y-chan/onnxruntime
`perf/gqa-split-decode-8bit-paged`).

Decoder step only (no embedding `Run`, no logits readback), CUDA graph,
block 16, a different token per request so the MoE routes the rows apart:

| requests | context | static fp16 / INT8 | paged fp16 | paged INT8 | paged FP8 |
|---|---|---|---|---|---|
| 1 | 750 | 3.89 / 3.93 ms | 3.98 ms | 4.01 ms | 4.02 ms |
| 8 | 750 | 11.9 / 12.2 ms | 11.3 ms | 11.3 ms | 11.3 ms |
| 16 | 750 | - / 18.3 ms | 17.6 ms | 17.1 ms | 17.3 ms |
| 32 | 750 | - | 23.7 ms | 22.8 ms | 22.9 ms |
| 8 | 3500 | 14.5 / 13.4 ms | 13.4 ms | 12.6 ms | 12.4 ms |
| 16 | 3500 | - / 20.7 ms | (pool full) | 21.0 ms | 20.1 ms |

At more than one request the step time depends on how far the rows' expert
choices overlap, so the static and paged columns differ by about as much as
two runs of either with different tokens. Prefill is the same within 3%
(747 tokens: 127 ms at block 16, 125 ms at block 256).

Memory (nvidia-smi, 24,576 MiB card): with a 40k-token fp16 pool (80k for
INT8 / FP8, the same 3,750 MiB) the loaded pipeline uses 22,916 MiB and 23,516
MiB after 32-request decode steps, leaving about 1,060 MiB for the audio
encoder's prefill workspace (600-850 MiB at 20 s of audio, arena shrinkage
on; not measured together with a full pool).

Teacher-forced on the same reference tokens (15 / 45 / 120 s clips), the paged
decoder against the static one gives a mean KL of 8e-4, top-1 agreement
99.6-100%; three requests decoded together against each alone, 3e-4 to 8e-4.
That is the model's own floor for fp16 arithmetic: the static decoder with the
split decode kernel against the regular Flash path measures 2.4e-4 to 9.2e-4.
In a single attention node, GQA and PagedAttention are both within fp16
rounding of an fp32 reference (cached K: 0.5 against 1 ulp).

