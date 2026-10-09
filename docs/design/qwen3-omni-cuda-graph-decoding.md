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

`--features gqa-cache`, alongside `static-cache`, hands the cache to
`com.microsoft::GroupQueryAttention` instead of `TensorScatter` + `Attention`.
It changes the decoder's I/O and is worth it only on some devices and prompt
lengths; the last section has the numbers.

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
pair GQA takes is computed inside the graph from it, so no other input
changes. A driving loop therefore needs new cache buffers and one fewer feed;
everything else about it is unchanged.

The feature is wired for speech-language exports (Qwen3-ASR,
Qwen3-forced-aligner, Qwen3-Omni). The causal-LM and Gemma4 static-cache paths
still emit `TensorScatter` + `Attention`; extending them is mechanical, and
Gemma4 additionally needs its bias layers to stay on the Attention op, which
is the one thing GQA cannot express.
