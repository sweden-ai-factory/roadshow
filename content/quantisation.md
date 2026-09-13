---
title: "Bonus: Quantisation"
teaching: 30
exercises: 15
---

# Quantisation

Large language models require substantial memory because they contain many
parameters. Reducing the numerical precision used to represent those
parameters can make a model easier to load, serve, and sometimes fine-tune.

This process is called **quantisation**.

A simple approximation of the memory needed to store model parameters is:

```{math}
\text{parameter memory}
=
\text{number of parameters}
\times
\text{bits per parameter}
```

For example, storing one billion parameters requires approximately:

```text
FP32:  4 GB
FP16:  2 GB
BF16:  2 GB
INT8:  1 GB
4-bit: 0.5 GB
```

These values describe parameter storage only. A working model also needs
memory for temporary computation, runtime buffers, cached attention states,
and framework overhead. Training introduces further requirements for
activations, gradients, and optimizer state.

Quantisation is related to resource-efficient fine-tuning, but it does not
solve the same problem as prompting, retrieval, or fine-tuning:

```text
Prompt engineering, RAG, and fine-tuning:
    change the information or behaviour of the system

Quantisation:
    changes how numerical values are represented and computed
```

A prompt-only model can be quantised. A model in a RAG pipeline can be
quantised. A fine-tuned model can also be quantised.

:::{questions}

- Why do numerical formats affect model memory?
- How do floating-point and integer representations differ?
- What information is lost during quantisation?
- Does using fewer bits always make a model faster?
- What is the difference between inference quantisation and QLoRA?
- Which parts of training memory are reduced by a quantised base model?
- How should a quantised model be evaluated?
:::

:::{objectives}

By the end of this episode, learners should be able to:

- estimate the memory needed to store model parameters at several
  precisions;
- explain quantisation as a mapping from a high-precision range to a smaller
  set of representable values;
- distinguish storage precision from computation precision;
- explain why lower bit width does not guarantee higher performance;
- distinguish post-training quantisation from quantisation-aware training;
- describe the main idea behind QLoRA;
- identify memory that is not removed by quantising the base model;
- design a comparison between a quantised model and its higher-precision
  baseline.
:::

## Why numerical precision matters

Neural-network parameters are numerical values. A numerical format determines
which values can be represented and how much memory is required to store each
one.

A model containing seven billion parameters requires approximately 28 GB to
store its weights in 32-bit precision:

```{math}
7 \times 10^9 \times 4\ \text{bytes}
\approx
28\ \text{GB}
```

The same parameter count requires approximately 14 GB when each parameter
occupies 16 bits:

```{math}
7 \times 10^9 \times 2\ \text{bytes}
\approx
14\ \text{GB}
```

An idealized 4-bit representation reduces the raw parameter storage to
approximately 3.5 GB:

```{math}
7 \times 10^9 \times 0.5\ \text{bytes}
\approx
3.5\ \text{GB}
```

Actual memory use will be higher. Quantised representations often require
additional scaling information, and the runtime allocates memory for other
purposes.

:::{admonition} Parameter size is not total memory use
:class: important

The number of parameters multiplied by the number of bytes per parameter is a
useful first estimate. It is not a complete estimate of the memory required to
run or train the model.

Inference may also require:

- key-value caches;
- temporary activations;
- dequantisation buffers;
- attention workspaces;
- runtime and framework overhead.

Training may additionally require:

- saved activations;
- gradients;
- optimizer states;
- trainable adapter parameters;
- communication buffers in distributed training.
:::

## Numerical formats

The names FP32, FP16, BF16, INT8, and 4-bit describe families of numerical
representations.

### FP32

FP32 is a 32-bit floating-point format. It provides a wide range and relatively
high precision, but requires four bytes per value.

FP32 was historically common for neural-network training. Modern accelerators
often achieve substantially higher throughput with 16-bit formats while still
retaining selected operations in FP32 where necessary for numerical
stability.

### FP16

FP16 uses 16 bits per value. Compared with FP32, FP16 reduces both numerical
range and precision.

The reduced range can cause overflow or underflow during training. Mixed
precision training may therefore use loss scaling and retain some values or
operations in FP32.

### BF16

BF16 also uses 16 bits, but allocates its bits differently from FP16. BF16 has
approximately the same numerical range as FP32 but less precision within that
range.

This makes BF16 attractive for training on supported accelerators. Its broad
range reduces some of the numerical difficulties associated with FP16.

FP16 and BF16 require the same number of bits, but they are not interchangeable
in terms of hardware support, numerical behaviour, or performance.

### Integer and low-bit representations

An 8-bit or 4-bit quantised representation provides far fewer possible bit
patterns than a 16-bit floating-point representation.

The model therefore cannot retain every original value exactly. Quantisation
maps a set of high-precision values onto a smaller set of representable
values.

For illustration, suppose a group of weights spans a range from \(-1\) to
\(1\). A high-precision representation can distinguish many values within this
range. A simple low-bit representation might approximate those values using a
small set of levels:

```text
original:   -0.91  -0.24   0.08   0.39   0.87
quantised:  -1.00  -0.33   0.00   0.33   1.00
```

The difference between an original value and the value used to represent it
is the quantisation error.

Real model-quantisation methods use more sophisticated mappings than this
example. The central trade-off remains the same:

```text
fewer representable values
    -> lower storage requirements
    -> some approximation error
```

## Scale and zero point

A common integer quantisation scheme maps a real-valued tensor \(x\) to an
integer tensor \(q\).

A simplified mapping is:

```{math}
q =
\operatorname{round}
\left(
\frac{x}{s}
\right)
+ z
```

where:

- \(s\) is the scale;
- \(z\) is the zero point;
- rounding selects the nearest representable integer.

An approximate real value can be reconstructed using:

```{math}
\hat{x} = s(q-z)
```

The quantised value \(q\) requires fewer bits than the original floating-point
value. The scale and zero point provide information needed to interpret it.

The reconstructed value \(\hat{x}\) is generally not exactly equal to \(x\).
The representation has deliberately traded numerical fidelity for lower
storage cost.

### Symmetric and asymmetric quantisation

In symmetric quantisation, the representable range is centered around zero.
The zero point is therefore zero or fixed by the representation.

In asymmetric quantisation, both a scale and a non-zero offset may be used.
This can represent distributions that are not centered around zero more
efficiently, at the cost of additional complexity.

Weights and activations do not necessarily use the same scheme. A method may
quantise weights symmetrically while treating activations differently.

## Quantisation granularity

A tensor does not need to share one scale across all of its values. The scale
can be calculated at different levels of granularity.

### Per-tensor quantisation

Per-tensor quantisation uses one scale for the complete tensor.

This keeps the representation simple, but a few unusually large or small
values can determine the range for every other value. The available levels may
then be used inefficiently for most of the tensor.

### Per-channel quantisation

Per-channel quantisation uses a separate scale for each output channel or
another selected dimension.

This can better represent channels whose weights have different
distributions, at the cost of storing more scaling information and using a
more complicated kernel.

### Group-wise quantisation

Group-wise quantisation divides weights into smaller groups and calculates a
scale for each group.

This is a common compromise for low-bit model weights. Smaller groups usually
reduce approximation error because each scale describes a more local range of
values. They also require more metadata and may affect kernel efficiency.

The term "4-bit model" therefore does not fully describe a quantised model.
The result also depends on the quantisation scheme, group size, scaling
method, treatment of outliers, and computation format.

## Weight and activation quantisation

A quantisation method may target different parts of the computation.

### Weight-only quantisation

Weight-only quantisation stores model weights in a lower-bit format.
Activations may remain in FP16, BF16, or another floating-point format.

This can reduce the memory needed to load the model and the memory bandwidth
needed to read its parameters. During computation, a kernel may dequantise
weight blocks into a computation format or incorporate the scaling directly
into the matrix multiplication.

Weight-only quantisation is common for large language model inference because
the model weights account for a substantial part of the memory footprint.

### Weight and activation quantisation

Other methods quantise both weights and activations.

This can enable lower-precision arithmetic throughout more of the computation,
but activation distributions vary with the input and can contain outliers.
Maintaining model quality can therefore be more difficult than for
weight-only quantisation.

### The key-value cache

Autoregressive generation stores keys and values from previous tokens so that
the model does not recompute the complete sequence at every generation step.
This storage is called the **key-value cache**, or KV cache.

The KV cache grows with sequence length, batch size, number of layers, and the
relevant attention dimensions. For long contexts or large batches, it may
consume a substantial fraction of inference memory.

Quantising model weights does not automatically quantise the KV cache. Some
inference systems support lower-precision KV caches as a separate
optimization.

:::{admonition} Identify what has been quantised
:class: note

When a library describes a model as "8-bit" or "4-bit", check whether the
description refers to:

- model weights;
- activations;
- the key-value cache;
- selected layers only;
- storage on disk;
- storage in accelerator memory;
- arithmetic performed by the accelerator.

These are related but distinct choices.
:::

## Post-training quantisation

Post-training quantisation starts from an already trained model and converts
some of its values into a lower-precision representation.

No further task-specific training is necessarily required.

A simplified workflow is:

```text
higher-precision trained model
    -> inspect or calibrate numerical distributions
    -> choose scales and quantisation parameters
    -> store quantised model
    -> evaluate quality and performance
```

Some post-training methods use a calibration dataset to observe representative
activations or estimate which weights are most sensitive to approximation.
The calibration data does not need to be as large as the original training
corpus, but it should reflect expected use.

Other methods can quantise weights without an explicit calibration dataset.
The resulting quality depends on the method, model architecture, and target
bit width.

Post-training quantisation is attractive because it does not require full
retraining. It is frequently considered when a model already behaves
satisfactorily but is too large or expensive to deploy in its original
precision.

## Quantisation-aware training

Quantisation-aware training simulates low-precision effects during training.
The optimization process can then adjust the model parameters to compensate
for some of the approximation introduced by quantisation.

A simplified view is:

```text
forward pass:
    simulate quantised values

backward pass:
    update higher-precision trainable parameters

deployment:
    convert to the intended low-precision representation
```

Quantisation-aware training generally involves more work than post-training
quantisation, but it can preserve quality better for demanding low-bit
configurations.

This should not be confused with QLoRA. Quantisation-aware training prepares a
model to operate after quantisation. QLoRA uses a quantised frozen base model
to reduce the memory required for parameter-efficient fine-tuning.

## Does quantisation make a model faster?

Not necessarily.

Fewer bits reduce the storage occupied by each value and may reduce the amount
of memory transferred when reading weights. This can improve performance when
inference is limited by memory bandwidth.

The accelerator still needs a suitable execution path. Performance depends
on whether optimized kernels exist for the numerical format, quantisation
scheme, group size, model architecture, and hardware.

A quantised operation may need to:

```text
read low-bit weights
    -> unpack values
    -> apply scales
    -> convert or dequantise values
    -> perform matrix multiplication
```

If the software does not implement these steps efficiently, a smaller model
may run no faster and can sometimes run more slowly than a higher-precision
model.

Small batches, short prompts, tokenizer overhead, CPU-to-device transfer, and
framework overhead may also dominate the total runtime. Quantising the weights
does not remove those costs.

:::{admonition} Measure on the intended system
:class: important

Do not infer speed from model size alone.

Benchmark the complete application on the intended hardware and software
stack. Measure at least:

- peak memory use;
- prompt-processing throughput;
- generation throughput;
- latency at realistic batch sizes;
- output quality on representative examples.
:::

## Quantisation and model quality

Quantisation replaces some parameter values with approximations. It can
therefore affect model output.

The effect is not uniform across models or tasks. A method that preserves
quality for common conversational prompts may affect numerical reasoning,
code generation, multilingual text, or specialist terminology differently.

The effect may also be hidden by open-ended evaluation. Two responses can use
different wording while both appearing plausible. A structured evaluation is
needed to determine whether the quantised model remains suitable.

A useful comparison keeps everything except precision constant:

```text
same model revision
same tokenizer
same prompt
same decoding configuration
same evaluation examples
different numerical representation
```

For deterministic comparisons, greedy decoding or a fixed random seed can
reduce variation. Application-level evaluation remains more important than
token-for-token equality.

## Quantisation during fine-tuning

Fine-tuning requires much more memory than inference because training needs
additional state.

A rough decomposition is:

```text
frozen or trainable model weights
trainable adapter weights
gradients
optimizer states
saved activations
temporary buffers
```

Full fine-tuning in a low-bit integer representation is not as simple as
storing every trainable parameter in four bits. Gradient-based optimization
requires sufficient numerical precision to accumulate useful updates.

Resource-efficient fine-tuning therefore often combines different numerical
formats rather than using one format for everything.

## QLoRA

QLoRA combines a quantised frozen base model with trainable LoRA adapters.

The base model is stored in a low-bit representation. The LoRA parameters,
gradients, optimizer state, and selected computations use higher precision
where required.

Conceptually:

```text
quantised frozen base model
             +
higher-precision trainable LoRA adapters
             =
memory-efficient parameter adaptation
```

The base weights do not need gradients because they are frozen. Quantising
those weights can therefore substantially reduce the memory needed to keep the
base model on the accelerator.

During a forward pass, the model computes both the frozen base-layer
contribution and the trainable adapter contribution:

```{math}
y = W_qx + \frac{\alpha}{r}BAx
```

Here, \(W_q\) denotes the quantised representation of the frozen base weight.
The matrices \(A\) and \(B\) are the trainable LoRA parameters.

The implementation may dequantise blocks of \(W_q\) as part of a specialized
matrix-multiplication kernel. Quantised storage does not imply that every
arithmetic operation is performed directly in the same low-bit format.

### What QLoRA saves

QLoRA primarily reduces the memory occupied by the frozen base-model weights.

LoRA already avoids storing gradients and optimizer states for those frozen
weights. Quantisation further reduces their storage.

This can make it possible to fine-tune a model on a device that cannot hold
the base model in FP16 or BF16 together with the rest of the training state.

### What QLoRA does not save

QLoRA does not remove activation memory. The forward pass must still process
the hidden states through every model layer.

Long sequences and large per-device batches can therefore exhaust memory even
when the base weights occupy relatively little space.

QLoRA also retains:

- adapter parameters;
- adapter gradients;
- adapter optimizer states;
- temporary dequantisation or computation buffers;
- framework and communication overhead.

If a training run uses little memory for the weights but substantial memory
for activations, reducing the weight precision further may have limited
effect. Activation checkpointing, shorter sequences, smaller batches, and
memory-efficient attention may then be more relevant.

:::{admonition} QLoRA is not ordinary full-model 4-bit training
:class: caution

The base model is quantised and frozen. The adapter is trained using
higher-precision computation where needed.

Describing QLoRA simply as "training the model in 4-bit" hides this important
division.
:::

## Compute dtype and storage dtype

A QLoRA configuration commonly distinguishes between the format used to store
the quantised weights and the format used for computation.

For example:

```text
base-weight storage:    4-bit quantised values
matrix computation:     BF16
LoRA parameters:        BF16 or FP32
optimizer state:        optimizer-dependent
```

The precise combination depends on the library and hardware.

A low-bit storage format reduces memory, while a supported compute dtype allows
the matrix operations to run efficiently and with adequate numerical range.

Choosing BF16 as a compute dtype does not turn the quantised base weights back
into permanently stored BF16 weights. Blocks can be converted as part of the
operation and discarded when no longer needed.

## Hardware and software support

Quantisation support is not determined only by the accelerator's ability to
represent integers.

An end-to-end workflow depends on several components:

```text
model architecture
    -> quantisation library
    -> framework integration
    -> low-level kernels
    -> accelerator runtime
    -> physical hardware
```

A configuration may be supported on one accelerator family but unavailable or
slow on another because the required kernels have not been implemented or
optimized.

Support can also differ between inference and training. A runtime may load a
quantised model for inference without supporting backward propagation through
the operations required by QLoRA.

Before selecting a quantisation method, verify:

- that the required library supports the accelerator;
- that the selected model layers can be quantised;
- that suitable kernels exist for the bit width and group size;
- that the intended compute dtype is supported;
- that backward propagation works if training is required;
- that distributed execution is supported if multiple devices will be used.

A successful model load is not sufficient evidence of an efficient
configuration. The application should be profiled under a realistic workload.

## Choosing whether to quantise

Quantisation is useful when model-weight memory is a meaningful constraint.

For inference, this may be the case when the model does not fit on the
available accelerator, when several model replicas are needed, or when memory
bandwidth limits throughput.

For fine-tuning, quantising a frozen base model can create enough memory for
LoRA training. This is particularly useful when the model is too large to load
in 16-bit precision together with activations and adapter training state.

Quantisation may not be the first optimization when the bottleneck lies
elsewhere. If the workload is dominated by long-sequence activations, KV-cache
growth, data loading, communication, or inefficient batching, lowering the
weight precision may not address the main problem.

A practical process is:

```text
measure the baseline
    -> identify the largest memory and runtime costs
    -> choose a supported quantisation scheme
    -> measure memory and performance
    -> evaluate output quality
    -> retain the quantised model only if the trade-off is acceptable
```

## Exercise: Estimate parameter memory

:::{exercise} Estimate model-weight storage
:label: exercise-quantisation-memory

Estimate the raw memory required to store the weights of a model containing
seven billion parameters using:

1. FP32;
2. FP16 or BF16;
3. INT8;
4. an idealized 4-bit representation.

Use decimal gigabytes, where one gigabyte is \(10^9\) bytes.

Why will the observed device memory be higher than these estimates?
:::

:::{solution} exercise-quantisation-memory
:class: dropdown

FP32 uses four bytes per parameter:

```{math}
7 \times 10^9 \times 4
=
28 \times 10^9\ \text{bytes}
=
28\ \text{GB}
```

FP16 and BF16 use two bytes per parameter:

```{math}
7 \times 10^9 \times 2
=
14\ \text{GB}
```

INT8 uses one byte per parameter:

```{math}
7 \times 10^9 \times 1
=
7\ \text{GB}
```

An idealized 4-bit representation uses half a byte per parameter:

```{math}
7 \times 10^9 \times 0.5
=
3.5\ \text{GB}
```

Observed memory will be higher because the runtime also stores quantisation
metadata, temporary buffers, activations, the KV cache during generation, and
framework state. Allocators may reserve additional device memory that is not
currently occupied by live tensors.
:::

## Exercise: Quantise a small set of values

:::{exercise} Apply a simple symmetric quantiser
:label: exercise-simple-quantiser

Suppose a simplified signed quantiser can represent the integer values:

```text
-3, -2, -1, 0, 1, 2, 3
```

Use a scale of:

```text
s = 0.25
```

Quantise the following values using:

```{math}
q = \operatorname{round}\left(\frac{x}{s}\right)
```

Clamp results to the representable integer range, then reconstruct the
approximations using:

```{math}
\hat{x} = sq
```

The original values are:

```text
-0.90, -0.38, 0.06, 0.44, 1.10
```

Which value has the largest absolute quantisation error?
:::

:::{solution} exercise-simple-quantiser
:class: dropdown

For \(-0.90\):

```text
-0.90 / 0.25 = -3.6
round and clamp -> -3
reconstructed -> -0.75
absolute error -> 0.15
```

For \(-0.38\):

```text
-0.38 / 0.25 = -1.52
round -> -2
reconstructed -> -0.50
absolute error -> 0.12
```

For \(0.06\):

```text
0.06 / 0.25 = 0.24
round -> 0
reconstructed -> 0.00
absolute error -> 0.06
```

For \(0.44\):

```text
0.44 / 0.25 = 1.76
round -> 2
reconstructed -> 0.50
absolute error -> 0.06
```

For \(1.10\):

```text
1.10 / 0.25 = 4.4
round -> 4
clamp -> 3
reconstructed -> 0.75
absolute error -> 0.35
```

The value \(1.10\) has the largest absolute error because it lies outside the
range represented by the selected scale and integer limits. This illustrates
the effect of outliers: expanding the range to include them may reduce
clipping but leave fewer useful levels for values near zero.
:::

## Exercise: Diagnose a memory bottleneck

:::{exercise} Will QLoRA solve the problem?
:label: exercise-qlora-bottleneck

A LoRA training run loads its frozen base model in a 4-bit representation.
Device memory use remains high and the run fails when the maximum sequence
length is increased from 1024 to 8192 tokens.

The per-device batch size is unchanged.

Answer the following questions:

1. Why does quantising the base model not prevent this failure?
2. Which category of memory is likely to have grown?
3. What changes could reduce memory use?
4. Which changes alter the effective training data or optimization behaviour?
:::

:::{solution} exercise-qlora-bottleneck
:class: dropdown

Quantising the base model reduces weight storage, but the forward pass still
creates and saves activations needed for backpropagation. Increasing the
sequence length greatly increases activation memory. Standard dense attention
also constructs objects whose dimensions depend quadratically on sequence
length.

Likely remedies include reducing the per-device batch size, using shorter
sequences, enabling activation checkpointing, using a memory-efficient
attention implementation, or changing how long examples are sampled and
packed.

Reducing sequence length may truncate training examples and therefore changes
the information available to the model. Reducing batch size changes the
microbatch and may require different gradient accumulation to maintain the
same effective batch size. Activation checkpointing primarily trades
additional computation for lower memory. Changing packing affects how
efficiently short examples use the available sequence length and must preserve
the intended example boundaries and labels.
:::

## Exercise: Design a quantisation benchmark

:::{exercise} Compare two model representations
:label: exercise-quantisation-benchmark

A team wants to replace a BF16 model with a 4-bit weight-quantised version.

Design a minimal comparison that determines whether the replacement is
worthwhile. Include memory, performance, and quality measurements.

Explain why testing one short prompt is insufficient.
:::

:::{solution} exercise-quantisation-benchmark
:class: dropdown

The comparison should keep the model revision, tokenizer, prompt templates,
decoding settings, and hardware fixed.

Memory measurements should include peak device memory for realistic prompt
lengths, output lengths, and batch sizes. The team should distinguish loaded
weight memory from memory consumed by the KV cache and temporary buffers.

Performance measurements should include prompt-processing throughput,
generation throughput, and end-to-end latency. Measurements should be
repeated for representative batch sizes and context lengths because the
bottleneck may change with the workload.

Quality should be evaluated on held-out examples from the intended
application. The evaluation should include any especially sensitive
capabilities, such as structured output, numerical reasoning, code,
multilingual input, or specialist vocabulary.

One short prompt cannot represent the range of sequence lengths, input types,
or model capabilities used by the application. Timing one request is also
sensitive to initial compilation, cache warm-up, and random system variation.
:::

## Summary

Quantisation reduces the number of bits used to represent model values. This
can substantially reduce parameter storage and may improve inference
performance when suitable kernels and hardware support are available.

Lower precision introduces approximation error. The effect depends on more
than the nominal bit width. Scaling method, group size, treatment of outliers,
quantised components, computation dtype, model architecture, and runtime
implementation all matter.

Weight-only quantisation reduces model-weight memory but does not
automatically reduce activation or KV-cache memory. Quantised models are not
automatically faster because unpacking, scaling, and dequantisation can add
work.

QLoRA combines a quantised frozen base model with trainable LoRA adapters. It
reduces the storage required for the base weights but retains activation
memory, adapter gradients, optimizer state, and temporary computation
buffers.

Quantisation should be introduced in response to a measured resource
constraint:

```text
measure
    -> identify the bottleneck
    -> quantise using a supported method
    -> benchmark again
    -> evaluate quality
```

The main principle is:

```text
A smaller numerical representation is useful only when it improves the
resource trade-off without making the model unsuitable for its task.
```
