---
title: Fine-tuning large language models
teaching: 45
exercises: 20
---

# Fine-tuning large language models

Fine-tuning continues the training of a pretrained model using examples
selected for a particular task or behaviour.

Unlike prompt engineering and retrieval-augmented generation, fine-tuning
changes model parameters. The resulting behaviour is encoded in a new set of
weights or in a smaller set of adapter parameters.

Fine-tuning can be useful, but it introduces a complete training lifecycle.
The project needs suitable data, an objective, a validation strategy,
hyperparameters, experiment tracking, model artifacts, and an evaluation that
goes beyond training loss.

:::{questions}

- What changes during fine-tuning?
- What kinds of behaviour can fine-tuning teach?
- How does full fine-tuning differ from parameter-efficient fine-tuning?
- What is low-rank adaptation?
- Which transformer layers can receive LoRA adapters?
- What must be prepared before training begins?
- How do we determine whether fine-tuning improved the application?

:::

:::{objectives}

By the end of this episode, learners should be able to:

- explain how fine-tuning differs from prompting and retrieval;
- identify tasks for which fine-tuning is a reasonable intervention;
- describe the practical costs of full fine-tuning;
- explain the idea behind parameter-efficient fine-tuning;
- describe LoRA as a low-rank update to frozen model weights;
- explain why LoRA target modules depend on model architecture;
- identify the data and evaluation requirements for a fine-tuning project;
- distinguish training progress from evidence of generalization.
:::

## What fine-tuning changes

A pretrained language model contains parameters learned from a large
collection of training data. During fine-tuning, the model processes examples
that represent a more specific objective. Gradients computed from those
examples update some or all of the model parameters.

For supervised fine-tuning, an example commonly contains an input and a
desired response:

```text
Input:
Summarize the incident report using the required internal schema.

Desired response:
{
  "category": "resource_exhaustion",
  "component": "gpu",
  "action": "reduce_batch_size"
}
```

The model predicts tokens in the desired response. The training objective
measures how much probability the model assigned to the expected tokens.
Backpropagation computes parameter updates that make those tokens more likely
for similar inputs.

The training process does not insert a symbolic rule such as:

```text
if the report mentions GPU memory:
    return resource_exhaustion
```

Instead, it adjusts a large number of numerical parameters. The resulting
behaviour must therefore be tested empirically.

### Behavioural adaptation

Fine-tuning is well suited to stable patterns that occur repeatedly.

A project might use fine-tuning to improve adherence to a specialized output
format, apply domain-specific labels, produce a consistent response style, or
map recurring inputs to recurring outputs.

The justification becomes stronger when the application will process many
examples. Including several demonstrations in every prompt consumes context
and inference time. A trained adaptation may encode the same pattern more
compactly.

### Fine-tuning and knowledge

Fine-tuning can change the factual associations produced by a model, but this
does not make it a suitable document database.

A fine-tuned model cannot reliably report which training document supports a
statement. It may combine the new examples with associations learned during
pretraining. Updating or removing a fact requires another model update, and
the effect of that update may not be isolated.

Changing and citable knowledge should normally remain in an external source
that can be retrieved at inference time.

A useful division of labour is:

```text
Fine-tuning:
    teaches stable behaviour

Retrieval:
    supplies current evidence
```

## Full fine-tuning

In full fine-tuning, every trainable parameter in the model is updated.

For a model with billions of parameters, this creates substantial resource
requirements. The device must hold the model weights, gradients, optimizer
state, activations, and temporary buffers used by the training operations.

The optimizer state can be particularly expensive. An optimizer may maintain
multiple values for every trainable parameter. Training therefore requires
considerably more memory than loading the same model for inference.

Full fine-tuning also produces a complete new set of model weights. If several
tasks require separate models, storage and deployment costs grow quickly.

### When full fine-tuning may be justified

Full fine-tuning offers the greatest flexibility because updates are not
restricted to selected components. It may be appropriate when the task
requires broad changes, the training dataset is sufficiently large, and the
project has the compute and evaluation capacity to manage the resulting
model.

These conditions are demanding. For many organizational and workshop use
cases, parameter-efficient fine-tuning is the more practical starting point.

:::{admonition} More trainable parameters are not automatically better
:class: caution

Updating the whole model increases capacity, but it also increases memory use,
storage, and the risk of changing useful existing behaviour.

The choice between full and parameter-efficient fine-tuning should be based
on measured task performance rather than on the assumption that a larger
update must be superior.
:::

## Parameter-efficient fine-tuning

Parameter-efficient fine-tuning, or PEFT, keeps most pretrained parameters
frozen and learns a much smaller set of task-specific parameters.

The base model still participates in every forward pass. Freezing the base
model does not remove the need to load it or execute its layers. The main
training savings come from not storing gradients and optimizer states for all
base-model parameters.

PEFT methods differ in where the trainable parameters are added and how they
interact with the frozen model. Low-rank adaptation is one of the most widely
used approaches.

## Low-rank adaptation

Suppose a pretrained layer contains a weight matrix:

```{math}
W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}
```

Full fine-tuning would learn an unrestricted update of the same shape:

```{math}
W' = W + \Delta W
```

LoRA represents the update as the product of two smaller matrices:

```{math}
\Delta W = BA
```

where:

```{math}
A \in \mathbb{R}^{r \times d_{\mathrm{in}}}
```

and:

```{math}
B \in \mathbb{R}^{d_{\mathrm{out}} \times r}
```

The rank \(r\) is chosen to be much smaller than the input and output
dimensions.

The adapted computation can be written as:

```{math}
y = Wx + \frac{\alpha}{r}BAx
```

The original matrix \(W\) remains frozen. Only \(A\) and \(B\) are trained.

Conceptually:

```text
large frozen weight matrix
           +
small trainable low-rank update
           =
adapted layer
```

### Why does a low-rank update help?

Fine-tuning does not necessarily require an independent change in every
direction of a large weight matrix. For many adaptation tasks, a lower
dimensional update can provide enough capacity to shift the model's
behaviour.

LoRA makes this assumption explicit by restricting the update to a matrix
whose rank is at most \(r\).

This reduces the number of trainable parameters from:

```{math}
d_{\mathrm{out}}d_{\mathrm{in}}
```

to:

```{math}
r(d_{\mathrm{in}} + d_{\mathrm{out}})
```

For large projection matrices and a small rank, the difference can be
substantial.

### Rank

The LoRA rank controls the maximum rank of the learned update. A higher rank
provides more adaptation capacity but adds trainable parameters and optimizer
state.

There is no universally correct rank. A small and well-defined task may need
little capacity. A broader adaptation may benefit from a larger rank or from
adapters on more modules.

Rank should be treated as a hyperparameter and evaluated on held-out data.

### Alpha

LoRA commonly scales the adapter contribution by:

```{math}
\frac{\alpha}{r}
```

The parameter \(\alpha\) controls the scale of the learned update relative to
the frozen base-layer output.

Changing rank without considering the scaling changes the effective size of
the adapter contribution. Libraries may offer different scaling conventions,
so configurations should be interpreted together with the implementation
used.

### Dropout

LoRA dropout applies dropout to the adapter path during training. It may reduce
overfitting, particularly for small datasets, but it is not automatically
beneficial.

A configuration with no dropout can be a reasonable baseline. If training
performance improves while validation performance does not, dropout is one
possible regularization measure to evaluate.

:::{admonition} Hyperparameters are empirical choices
:class: important

Rank, alpha, dropout, learning rate, batch size, sequence length, and training
duration should not be copied from a tutorial without checking whether the
tutorial uses a comparable model, dataset, and objective.

Begin with a defensible baseline, record the configuration, and compare
changes on held-out examples.
:::

## Where LoRA adapters are placed

LoRA is applied to linear transformations in the model. Attention projections
are common targets:

```text
query projection
key projection
value projection
attention output projection
```

Many implementations name these modules using forms such as:

```text
q_proj
k_proj
v_proj
o_proj
```

The names are conventions, not a universal interface.

Adapters may also be placed on feed-forward projections. A gated
feed-forward network may contain modules with names resembling:

```text
gate_proj
up_proj
down_proj
```

Targeting more modules gives the adapter more ways to modify the model. It
also increases the number of trainable parameters and the computational cost
of the adapter path.

### Why attention projections are common targets

Attention projections determine how token representations are converted into
queries, keys, values, and attention outputs. Adapting those projections can
change which contextual relationships the model emphasizes and how
information is exchanged between positions.

This makes attention projections a natural target for tasks that require a
change in how the model responds to context.

Feed-forward layers also contain substantial capacity. Including their
projections can improve adaptation for some tasks, although the additional
parameters should be justified through evaluation.

### Finding the correct module names

The correct target names must be obtained from the model implementation.

A reliable procedure is:

1. inspect the printed model architecture;
2. enumerate named modules;
3. identify linear layers within repeated transformer blocks;
4. check model-specific examples and documentation;
5. create the adapter;
6. verify which parameters are marked as trainable.

A simplified inspection pattern is:

```python
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)
```

The output should be inspected before choosing target modules. Copying
`q_proj` from another model is not sufficient if the current implementation
uses different names or a combined query-key-value projection.

:::{admonition} Verify the trainable parameters
:class: caution

After applying LoRA, inspect the model's trainable parameter count and names.

An unexpectedly small count may indicate that important modules were omitted.
An unexpectedly large count may indicate that the base model was not frozen
or that more modules were targeted than intended.
:::

## Preparing fine-tuning data

A fine-tuning dataset should represent the behaviour expected in deployment.

For an instruction-tuned chat model, examples are often represented as
messages:

```python
{
    "messages": [
        {
            "role": "user",
            "content": "Explain why this training run failed."
        },
        {
            "role": "assistant",
            "content": "The run exhausted device memory while allocating..."
        }
    ]
}
```

The model's chat template converts the messages into the control tokens and
text format expected by that model.

Using the correct template matters. A model trained with one conversation
format may not interpret another model's role markers as intended.

### Representative data

Representative data covers the variation expected in real inputs. It should
include routine examples, difficult cases, ambiguous cases, long and short
inputs, and examples where the correct response is to refuse or state that
information is missing.

Repeatedly generating examples from one template may produce a large dataset
without producing meaningful diversity.

Quality is also more important than raw example count. Incorrect answers,
inconsistent formatting, duplicated examples, and accidental sensitive data
can all become part of the learned behaviour.

### Training, validation, and test data

Training examples are used to update model parameters. Validation examples are
used during development to observe generalization and compare configurations.
A final test set should remain separate until the main decisions have been
made.

The splits must avoid leakage. Near-duplicate examples or several records from
the same original document can make validation results appear stronger than
they are.

For grouped data, split by the appropriate unit rather than by individual
row. If several examples come from one incident or one source document, all
of them may need to remain in the same split.

### What should contribute to the loss?

For assistant-response training, the input conversation provides context while
the desired assistant response provides the prediction target.

A common approach masks user and system tokens in the labels:

```text
system message       -> context, ignored by loss
user message         -> context, ignored by loss
assistant response   -> context and prediction target
```

The input is still needed because the assistant response must be conditioned
on the request. Masking a token from the loss does not remove it from the
model input.

Different objectives may use different masking strategies. The important
point is to decide explicitly which tokens the model is being trained to
predict.

## Training configuration

A training configuration controls both optimization and resource use.

The most visible parameters include learning rate, number of epochs, batch
size, gradient accumulation, maximum sequence length, evaluation frequency,
and checkpoint frequency.

These quantities interact. Changing one may alter memory use, runtime, or the
number of parameter updates.

### Batch size and gradient accumulation

The per-device batch size is the number of examples processed by one device
in one forward and backward pass.

Gradient accumulation performs several such passes before applying an
optimizer step. With data-parallel training, the effective batch size is:

```{math}
B_{\mathrm{effective}}
=
B_{\mathrm{device}}
\times
N_{\mathrm{devices}}
\times
N_{\mathrm{accumulation}}
```

For example:

```text
per-device batch size       4
number of devices           2
gradient accumulation       8
effective batch size       64
```

Gradient accumulation can produce a larger effective batch without storing
all examples' activations simultaneously. It does not avoid their
computation. Eight accumulation passes still require eight forward and
backward passes.

### Sequence length

Sequence length has a strong effect on memory and runtime. Longer sequences
require more activation memory, and standard dense attention forms a
quadratic attention matrix.

A batch size that works for short examples may fail when the batch contains
long examples.

Setting a maximum sequence length requires a data decision as well as a
resource decision. Truncation can remove parts of the prompt or desired
answer. Before choosing the limit, inspect the token-length distribution of
the dataset and decide which portion of each example may be truncated.

### Packing

Packing places several short examples into one fixed-length training
sequence. This reduces computation wasted on padding.

Without packing:

```text
example 1 + padding + padding + padding
example 2 + padding + padding + padding
```

With packing:

```text
example 1 + example 2 + example 3 + remaining padding
```

The dataloader batch size still refers to packed sequences, not to the number
of original examples contained inside them. One packed sequence may contain
several short examples.

Packing can improve throughput for datasets with many short records. It does
not increase the configured context length, and examples must remain properly
separated so that labels and attention behaviour follow the intended training
objective.

### Learning rate and warmup

The learning rate controls the scale of optimizer updates. An excessively
large learning rate may destabilize training or damage useful pretrained
behaviour. An excessively small learning rate may produce little adaptation
within the available training time.

Warmup starts with a smaller learning rate and increases it during the early
part of training. This can reduce instability before the optimizer has
established useful parameter statistics.

The appropriate values depend on the optimizer, effective batch size, model,
adapter configuration, and dataset. Training loss and validation behaviour
should be observed together.

### Training duration

More epochs expose the model to the training examples more times. This may
improve learning initially, but eventually increases the risk of overfitting.

A small dataset can be memorized quickly. The best checkpoint is not
necessarily the final checkpoint.

Evaluate at useful intervals and retain enough checkpoint information to
compare validation results rather than relying only on the last training
step.

## Evaluation during fine-tuning

Training loss answers a narrow question: how well does the current model
predict the training targets?

It does not show whether the model generalizes to new examples, whether the
application has improved, or whether unrelated capabilities have degraded.

### Validation loss

Validation loss applies the same token-prediction objective to examples that
were not used for parameter updates.

If training loss continues to fall while validation loss stops improving or
begins to rise, overfitting is a possible explanation.

Validation loss is useful, but application-level evaluation remains necessary.
A small change in language-model loss may not correspond to a meaningful
change in task quality.

### Task evaluation

The evaluation should match the intended behaviour.

For structured output, measure whether the response parses and whether the
fields are correct. For classification, use suitable class-level metrics. For
summarization or open-ended generation, combine human review with checks for
required content, unsupported claims, format adherence, and omissions.

Compare the fine-tuned model with the strongest non-fine-tuned baseline, not
only with an unhelpfully vague prompt.

### Regression evaluation

Fine-tuning may improve the target task while changing other behaviour.

A regression set should cover important capabilities that the adapted model
must retain. This may include general instruction following, multilingual
inputs, code formatting, safety constraints, or appropriate abstention.

The contents of the regression set depend on the application. The important
point is to decide what must not worsen before accepting the adapter.

:::{admonition} Evaluation is part of fine-tuning
:class: important

A training run without held-out evaluation is an experiment in optimization,
not evidence that the application improved.
:::

## Saving and deploying adapters

One advantage of LoRA is that the adapter is much smaller than the full base
model.

The base model and adapter can be stored separately:

```text
base model
    +
task-specific adapter
    =
adapted model
```

Several adapters can share the same base model artifact. This simplifies
storage and makes the distinction between the original model and the
task-specific update explicit.

For deployment, the adapter may be loaded alongside the base model. Some
workflows also merge the adapter update into the base weights. Merging can
simplify inference in environments that do not support adapters directly, but
it produces another complete model artifact.

The deployment record should identify:

```text
base-model name and revision
tokenizer revision
adapter artifact
training-data version
training configuration
evaluation results
software environment
```

Without these details, reproducing or auditing the model becomes difficult.

## A minimal fine-tuning workflow

A fine-tuning project can be organized as the following sequence:

```text
define the target behaviour
    -> build and evaluate a prompt baseline
    -> collect representative examples
    -> create train, validation, and test splits
    -> select the base model and chat template
    -> select full fine-tuning or PEFT
    -> configure and run training
    -> inspect validation and checkpoints
    -> evaluate task performance and regressions
    -> package the model or adapter
    -> monitor behaviour after deployment
```

The training command occupies only one part of this sequence. Data definition
and evaluation usually determine whether the result is useful.

## Exercise: Count LoRA parameters

:::{exercise} Count the adapter parameters
:label: exercise-lora-parameters

A linear layer has:

```text
input dimension:   4096
output dimension:  4096
LoRA rank:            8
```

Calculate:

1. the number of parameters in the original weight matrix;
2. the number of trainable parameters in the two LoRA matrices;
3. the fraction of the original parameter count represented by the adapter.

Ignore biases.
:::

:::{solution} exercise-lora-parameters
:class: dropdown

The original matrix contains:

```{math}
4096 \times 4096 = 16{,}777{,}216
```

parameters.

The two LoRA matrices contain:

```{math}
8 \times 4096 + 4096 \times 8
=
65{,}536
```

parameters.

The ratio is:

```{math}
\frac{65{,}536}{16{,}777{,}216}
=
0.00390625
```

The adapter therefore contains approximately \(0.39\%\) as many parameters as
the original matrix.

This calculation concerns one adapted matrix. A complete configuration may
apply LoRA to many projections across all transformer blocks.
:::

## Exercise: Select LoRA targets

:::{exercise} Inspect a model before configuring LoRA
:label: exercise-lora-targets

Suppose a model contains repeated blocks with the following linear modules:

```text
self_attn.query_key_value
self_attn.dense
mlp.dense_h_to_4h
mlp.dense_4h_to_h
```

A tutorial for another model recommends:

```python
target_modules=["q_proj", "v_proj"]
```

Answer the following questions:

1. Why is the tutorial configuration unlikely to work as intended?
2. Which modules in the current model might correspond to attention
   projections?
3. What should be verified after applying the adapter?
4. What is the trade-off involved in also targeting the MLP projections?
:::

:::{solution} exercise-lora-targets
:class: dropdown

The tutorial uses module names from a different model implementation. The
current model has no modules named `q_proj` or `v_proj`, so the adapter library
may match nothing or raise an error.

The combined `self_attn.query_key_value` module appears to contain the query,
key, and value projections. The `self_attn.dense` module is likely the
attention output projection. The exact interpretation should be checked
against the model implementation.

After applying the adapter, inspect the names and number of trainable
parameters. Confirm that the base-model parameters are frozen and that LoRA
parameters were created for the intended modules.

Targeting the MLP projections gives the adapter more capacity to change the
model, but increases trainable parameters, optimizer state, and computation.
Whether that improves the task should be determined through held-out
evaluation.
:::

## Exercise: Review a proposed experiment

:::{exercise} Review a fine-tuning plan
:label: exercise-review-finetuning

A proposed experiment has the following design:

```text
Dataset:                 600 examples
Training split:          all 600 examples
Evaluation:              final training loss
Epochs:                  20
Base model:              selected because it is popular
LoRA targets:            copied from an unrelated tutorial
Prompt baseline:         none
Regression tests:        none
```

Identify the weaknesses in the plan and propose an improved sequence of work.
:::

:::{solution} exercise-review-finetuning
:class: dropdown

The plan has no held-out examples, so it cannot measure generalization. Final
training loss shows how well the model predicts examples on which it was
trained, not whether the application improved.

Twenty epochs may cause a small dataset to be memorized. The appropriate
training duration should be selected using validation behaviour and
task-level evaluation.

The base model should be chosen according to the task, license, supported
context length, language coverage, deployment environment, and baseline
performance.

LoRA targets must match the current model architecture. Named modules should
be inspected and trainable parameters verified after adapter creation.

The absence of a prompt baseline means there is no evidence that training is
needed and no meaningful comparison for the result. The absence of regression
tests means target-task improvement could conceal damage to other important
behaviour.

A better sequence is:

```text
define measurable success
    -> construct held-out evaluation examples
    -> evaluate a strong prompt baseline
    -> inspect and clean the dataset
    -> create leakage-resistant splits
    -> choose the base model using baseline results and constraints
    -> inspect module names and configure LoRA
    -> train for a modest duration with validation
    -> compare checkpoints on task and regression evaluations
```

:::

## Summary

Fine-tuning changes model parameters by continuing training on examples of a
target behaviour.

Full fine-tuning updates the complete model. It provides substantial capacity
but has high memory, storage, and evaluation costs.

Parameter-efficient fine-tuning keeps most pretrained parameters frozen.
LoRA represents selected weight updates as the product of two small matrices.
This greatly reduces the number of trainable parameters and the associated
optimizer state.

LoRA adapters are commonly applied to attention projections and may also be
applied to feed-forward projections. The correct target-module names depend
on the model implementation and must be inspected rather than copied
uncritically from another architecture.

The success of fine-tuning depends on more than the training configuration.
The project needs representative data, leakage-resistant splits, an explicit
loss objective, a strong prompt baseline, held-out task evaluation, and
regression tests.

The main principle is:

```text
Fine-tuning is justified by a measured behavioural improvement on new
examples, not by a decreasing training loss.
```
