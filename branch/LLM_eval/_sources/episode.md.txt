# Anatomy of an LLM

Large language models can be adapted to a task in several ways. The simplest
approach is to improve the instructions given to the model. More involved
approaches add external information at inference time or modify some of the
model parameters through training.

In this lesson, we will use the following progression:

```text
Prompt engineering -> Retrieval-augmented generation -> Fine-tuning
```

Moving from left to right gives us more control, but also introduces more
infrastructure, data requirements, evaluation work, and operational cost.

:::{admonition} Central question
:class: attention

What is the least complex intervention that produces sufficiently reliable
results for our application?
:::

:::{questions}

- How does a transformer turn text into predictions?
- What do attention and feed-forward layers contribute?
- What can prompt engineering change?
- When should external knowledge be supplied using retrieval?
- When is fine-tuning justified?
- Where do LoRA and quantization fit into this progression?

:::

:::{objectives}

By the end of this lesson, learners should be able to:

- describe the path from text to tokens, embeddings, contextual
  representations, and output predictions;
- distinguish encoder-only, decoder-only, and encoder-decoder transformers;
- explain the roles of queries, keys, values, causal masking, and
  feed-forward networks;
- improve a prompt by specifying direction, format, examples, and evaluation
  criteria;
- distinguish behavioral adaptation from knowledge retrieval;
- choose between prompt engineering, RAG, and fine-tuning for a concrete use
  case;
- explain why LoRA reduces the number of trainable parameters;
- describe quantization as a memory and deployment optimization;
- identify the evaluation and maintenance costs introduced at each level of
  adaptation.

:::

### Prerequisites

Learners should have:

- basic familiarity with Python;
- used a chat-based language model at least once;
- a conceptual understanding of vectors and matrix multiplication;
- no prior experience with model training.

A detailed derivation of backpropagation is not required.

---

# 1. How a language model processes text

Before deciding how to adapt a model, it helps to understand where model
behavior comes from.

A transformer does not operate directly on words or sentences. A simplified
forward pass is:

```text
text
  -> tokens
  -> token IDs
  -> embeddings
  -> transformer blocks
  -> output logits
  -> next-token probabilities
```

For a generative model, this process is repeated one token at a time.

## 1.1 Tokenization

A tokenizer divides text into units called **tokens** and maps each token to an
integer ID.

A vocabulary containing every possible word would be impractical. It would be
very large and would need separate entries for related forms such as
`compute`, `computer`, and `computing`.

Modern language models therefore commonly use **subword tokenization**.
Frequent words may remain intact, while rarer words are divided into reusable
pieces.

For example, a tokenizer might process:

```text
supercomputing
```

as something conceptually similar to:

```text
super + comput + ing
```

The actual split depends on the tokenizer.

:::{admonition} Why tokenization matters
:class: note

Tokenization affects:

- the effective length of an input;
- the cost of inference and training;
- how efficiently different languages and specialist vocabularies are
  represented;
- whether the model sees a domain-specific term as one familiar unit or as
  several rare fragments.
:::

## 1.2 Embeddings

Token IDs are discrete integers. Neural networks instead operate on vectors.

An embedding layer contains a learned matrix:

```text
number of tokens in the vocabulary x embedding dimension
```

Looking up a token ID selects one row from this matrix. That row is the token's
initial embedding.

Tokens used in similar contexts tend to acquire related representations.
However, the initial token embedding alone is not enough to represent the
meaning of a token in a particular sentence.

Consider the token `mole` in:

```text
A mole damaged the garden.
The chemist measured one mole of the compound.
She has a mole on her cheek.
```

The initial token embedding is the same in each sentence. Transformer layers
turn it into a different contextual representation in each case.

---

# 2. Transformer anatomy

A transformer is composed of repeated blocks. Each block contains two major
computational components:

1. a multi-head attention sublayer;
2. a position-wise feed-forward network, often called an MLP.

Residual connections and normalization layers stabilize and preserve the
flow of information through the model.

```{figure} figures/transformer-block.svg
:alt: Simplified transformer block containing attention, a feed-forward
      network, residual connections, and normalization.
:width: 80%

A simplified transformer block.
```

A useful conceptual model is:

```text
attention:
    move information between token positions

feed-forward network:
    transform the representation at each position
```

This distinction is important for understanding prompt engineering, RAG,
fine-tuning, and LoRA.

## 2.1 Transformer families

Transformer architectures are commonly grouped into three families.

### Encoder-only models

Encoder-only models produce representations using context from both sides of a
token. They are well suited to tasks such as:

- text classification;
- named-entity recognition;
- semantic similarity;
- extractive question answering.

### Decoder-only models

Decoder-only models predict the next token from the tokens preceding it. They
are the dominant architecture for generative large language models.

During training, causal masking prevents each position from attending to
future tokens.

### Encoder-decoder models

Encoder-decoder models first encode an input sequence and then generate an
output sequence. They are often used for:

- translation;
- summarization;
- other sequence-to-sequence tasks.

:::{admonition} Workshop focus
:class: note

This lesson concentrates on decoder-only models because they are typical of
modern text-generation systems and instruction-tuned chat models.
:::

## 2.2 Self-attention

Self-attention lets each token representation incorporate information from
other relevant positions.

For every token, an attention head constructs three vectors:

- a **query**, describing what information this position is looking for;
- a **key**, describing what information a position may provide;
- a **value**, containing the information that can be transferred.

These vectors are learned projections of the current token representation.

The scaled dot-product attention operation is:

```{math}
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\mathsf{T}}{\sqrt{d_k}}
\right)V
```

The calculation has three conceptual steps:

1. compare queries with keys;
2. normalize the resulting relevance scores;
3. use the scores to combine the value vectors.

The dot product between a query and a key is large when they align strongly.
Dividing by the square root of the key dimension keeps the scale numerically
manageable. Softmax converts the scores into weights that sum to one.

```{figure} figures/query-key-value.svg
:alt: Tokens projected into query, key, and value vectors, followed by an
      attention matrix and a weighted combination of values.
:width: 90%

Queries and keys determine where information comes from. Values determine
what information is transferred.
```

:::{admonition} Avoid an overly literal interpretation
:class: caution

It can be useful to imagine an attention head looking for grammatical
relationships, such as a noun attending to an adjective. In real models,
attention features are learned, distributed, and often more abstract than
human-interpretable grammatical rules.
:::

## 2.3 Causal masking

A decoder-only model must not use future tokens to predict an earlier token.

For the sequence:

```text
The cat sat down
```

the representation at `cat` may use `The`, but it must not use `sat` or `down`
when learning to predict the next token.

This is enforced by applying a causal mask to the attention-score matrix:

```text
             key position
             1  2  3  4
query 1      x  -  -  -
query 2      x  x  -  -
query 3      x  x  x  -
query 4      x  x  x  x
```

Here, `x` denotes an allowed attention connection and `-` a masked connection.

For standard dense attention, the attention matrix grows with the square of
the sequence length. Long contexts can therefore require substantial memory
and computation.

## 2.4 Multi-head attention

A model uses several attention heads in parallel. Each head has its own
query, key, and value projections.

Different heads can learn different patterns of information flow. Their
outputs are concatenated and projected back into the model's hidden
dimension.

Multi-head attention is not a set of manually assigned linguistic rules.
The heads learn useful projections from the training objective.

## 2.5 Feed-forward networks

After attention has exchanged information between positions, the feed-forward
network transforms each position independently.

A simplified feed-forward operation is:

```{math}
\operatorname{FFN}(x) = W_2 \sigma(W_1 x + b_1) + b_2
```

The same network is applied to every token position.

In many transformer architectures, the feed-forward layers contain a large
fraction of the model parameters. They provide substantial representational
capacity, while attention controls how information flows between positions.

:::{admonition} A careful interpretation of model knowledge
:class: important

It is common to say that factual associations are "stored" in feed-forward
layers. This is a helpful first approximation, but model knowledge is
distributed across embeddings, attention projections, MLP parameters, and
their interactions.

Hallucinations should consequently not be attributed to one single component.
They arise from the model's probabilistic objective, its training data, the
prompt and context, and the absence of reliable external evidence.
:::

## 2.6 Residual connections and normalization

Transformer blocks also contain residual connections and normalization.

Residual connections allow a block to update, rather than completely replace,
a representation:

```{math}
x_{\mathrm{out}} = x_{\mathrm{in}} + f(x_{\mathrm{in}})
```

This makes it easier to train deep networks and helps information move through
many layers.

Normalization keeps activation scales manageable and supports stable
optimization.

```{exercise} Trace information through a transformer
:label: exercise-transformer-trace

Consider the sentence:

> The researcher deposited the data because it was required by the journal.

Discuss the following questions in pairs:

1. Why is the initial embedding for `it` insufficient?
2. Which earlier tokens might provide useful contextual information?
3. What is the role of query-key matching?
4. What information is carried by the value vectors?
5. What changes in a decoder-only model if relevant information appears after
   `it`?
```

```{solution} exercise-transformer-trace
:class: dropdown

1. The initial embedding represents the token type but does not identify what
   `it` refers to in this sentence.
2. Tokens associated with `data`, `deposited`, `required`, and `journal` may be
   relevant.
3. Query-key matching determines which positions are relevant to updating the
   representation of `it`.
4. Value vectors carry the information combined into the updated
   representation.
5. Causal masking prevents `it` from using later positions while predicting
   the next token. A later representation may still integrate information
   from `it`, but information cannot flow backwards from a future position.
```

---

# 3. A ladder of adaptation

There are several ways to influence a language model.

```text
Level 1: Prompt engineering
         Change the input.

Level 2: Retrieval-augmented generation
         Change the input and supply external evidence.

Level 3: Fine-tuning
         Change some or all of the model parameters.
```

This ladder represents increasing complexity, not increasing quality in every
situation.

```{figure} figures/adaptation-ladder.svg
:alt: Three ascending levels labelled prompt engineering, retrieval-augmented
      generation, and fine-tuning, with increasing cost, infrastructure, and
      maintenance.
:width: 85%

The adaptation ladder. Start with the least complex level that can satisfy the
requirements.
```

As we move upward, we generally increase:

- engineering effort;
- infrastructure requirements;
- data preparation;
- evaluation requirements;
- maintenance cost;
- the number of ways the system can fail.

A practical development strategy is:

```text
define success criteria
    -> build a prompt baseline
    -> evaluate
    -> add retrieval if evidence is missing
    -> evaluate
    -> fine-tune if behavior remains inadequate
```

:::{admonition} Important
:class: attention

Do not fine-tune merely because it is technically possible.

Fine-tuning is worthwhile when repeated, measurable deficiencies remain after
a strong prompt and, where relevant, a retrieval baseline.
:::

---

# 4. Level 1: Prompt engineering

Prompt engineering is the practice of designing model inputs to improve the
relevance, accuracy, reliability, and controllability of the output.

Prompt engineering does not change model parameters. It changes the context
from which the model generates a response.

## 4.1 Five prompt-design principles

### Give direction

State the task and the intended audience.

Instead of:

```text
Explain transformers.
```

try:

```text
Explain self-attention to research software engineers who understand vectors
but have not studied deep learning.
```

### Specify the output format

Describe the expected structure.

```text
Return:

1. a two-sentence summary;
2. three key concepts as bullet points;
3. one limitation;
4. one concrete example.
```

### Provide examples

Examples are useful when the desired behavior is easier to demonstrate than
to describe.

```text
Input: GPU memory exhausted during model loading
Category: resource limitation

Input: Dataset field contains unexpected strings
Category: data-format error

Input: Collective operation times out
Category:
```

### Divide the work

Break a complex task into stages.

```text
1. Identify the claims in the text.
2. List the evidence provided for each claim.
3. Mark unsupported claims.
4. Produce a revised version containing only supported claims.
```

The stages may be contained in one prompt or implemented as a prompt chain.

### Evaluate quality

Prompt design is an empirical process. A prompt that works for one example may
fail across a broader dataset.

Test prompts on:

- typical inputs;
- difficult inputs;
- ambiguous inputs;
- long inputs;
- malformed inputs;
- cases where the correct response is to abstain.

## 4.2 Role prompting

Role prompting asks the model to adopt a functional perspective.

```text
Act as a technical editor for documentation aimed at research software
engineers.
```

A role can influence tone, vocabulary, and priorities, but it does not grant
the model new knowledge or guaranteed expertise.

A role should be combined with explicit instructions and acceptance criteria.

## 4.3 Prompt chaining

Prompt chaining connects several focused model calls.

```text
source material
    -> extract claims
    -> retrieve evidence
    -> compare claims with evidence
    -> write final answer
```

Chaining can make intermediate results inspectable and simplify evaluation.
It can also increase latency and create additional failure points.

## 4.4 Meta-prompting

A meta-prompt asks a model to formulate or improve instructions.

For example:

```text
Rewrite the following task as a precise prompt. Ask for any missing
information, specify the expected output schema, and include criteria for
rejecting unsupported conclusions.
```

Meta-prompting can accelerate prompt development, but the resulting prompt
still needs systematic evaluation.

```{exercise} Improve a prompt
:label: exercise-improve-prompt

Start with:

> Summarize this research paper.

Rewrite the prompt so that it specifies:

1. the intended reader;
2. the maximum length;
3. the required structure;
4. how limitations should be handled;
5. what the model should do if information is missing.
```

```{solution} exercise-improve-prompt
:class: dropdown

One possible prompt is:

> Summarize the supplied research paper for research software engineers who
> are familiar with machine learning but not with this application domain.
> Use at most 300 words. Structure the response under the headings Question,
> Method, Results, Limitations, and Reproducibility. Distinguish limitations
> reported by the authors from limitations you infer. Do not invent missing
> details. Write "not reported in the supplied text" when the source does not
> contain the required information.
```

## 4.5 When prompt engineering is enough

Prompt engineering is often enough when:

- the model already has the necessary capability;
- all required information fits in the context;
- the task can be described clearly;
- output variation is acceptable or can be evaluated;
- examples can adequately demonstrate the desired behavior;
- low implementation and maintenance cost are important.

Move beyond prompting when failures are caused by missing or changing
knowledge, rather than unclear instructions.

---

# 5. Level 2: Retrieval-augmented generation

Retrieval-augmented generation, or RAG, retrieves relevant external material
and adds it to the model's context.

A simplified RAG pipeline is:

```text
user question
    -> search or retrieval
    -> selected passages
    -> prompt containing question and evidence
    -> generated answer
```

RAG changes the evidence available at inference time. It does not normally
change the model parameters.

## 5.1 Why use RAG?

RAG is useful when answers depend on:

- private documents;
- scientific literature;
- frequently changing information;
- organizational policies;
- source attribution;
- evidence that should be inspectable by the user.

Because documents can be updated independently of the model, RAG is often a
better choice than fine-tuning for changing factual knowledge.

## 5.2 What RAG does not guarantee

RAG can reduce unsupported answers, but retrieval alone does not guarantee
correctness.

A RAG system can still fail because:

- the required document was not indexed;
- the query did not retrieve the relevant passage;
- chunk boundaries removed necessary context;
- the ranking placed irrelevant passages first;
- too much context distracted the model;
- the model ignored or misinterpreted the evidence;
- the source itself was incorrect.

RAG must therefore be evaluated as a full system.

## 5.3 Minimal grounded-generation prompt

A basic prompt for the generation stage could be:

```text
Answer the question using only the supplied sources.

If the sources do not contain enough information, state that the available
evidence is insufficient.

For every factual claim, cite the corresponding source identifier.

Question:
{question}

Sources:
{retrieved_passages}
```

This prompt is only one component. Retrieval quality and source management are
at least as important as prompt wording.

## 5.4 When RAG is worth the complexity

RAG is a strong candidate when:

- the problem is primarily missing knowledge;
- the knowledge changes more quickly than a model can be retrained;
- users need citations or source traceability;
- documents are private or organization-specific;
- the corpus can be indexed and searched effectively.

RAG may be unnecessary when the task is transformation rather than knowledge
retrieval, such as consistently rewriting text into a fixed style.

```{exercise} Diagnose a RAG system
:label: exercise-diagnose-rag

A user asks:

> What is our current policy for storing sensitive research data?

The system returns an outdated policy even though the current policy exists in
the document collection.

List at least four possible causes. For each cause, suggest one diagnostic
test.
```

```{solution} exercise-diagnose-rag
:class: dropdown

Possible causes and tests include:

- **The current document was not indexed.**
  Search the index directly for a unique sentence from the current policy.

- **Metadata filters excluded the current document.**
  Log and inspect the filters used for the request.

- **The query did not match the terminology in the current policy.**
  compare retrieval results for the original query and several paraphrases.

- **The current policy was split into poor chunks.**
  inspect the retrieved chunks and their neighboring chunks.

- **The ranking favored the older document.**
  inspect retrieval scores, timestamps, and reranking output.

- **Both documents were supplied and the model chose the older one.**
  log the complete generation context and require the prompt to prioritize the
  policy with the most recent effective date.
```

---

# 6. Level 3: Fine-tuning

Fine-tuning continues training a pretrained model using examples chosen for a
particular objective.

Unlike prompt engineering and RAG, fine-tuning changes model parameters.

Fine-tuning is particularly useful for repeated behavioral patterns, such as:

- generating a specialized output structure;
- following domain-specific conventions;
- using specialist vocabulary;
- adopting a consistent response style;
- mapping recurring inputs to recurring outputs;
- improving behavior on a well-defined task with representative examples.

## 6.1 Fine-tuning is not a document database

Fine-tuning should not be the default way to insert changing factual
information.

A fine-tuned model:

- does not provide a reliable mechanism for exact factual recall;
- cannot easily show where a learned statement came from;
- may mix new examples with prior associations;
- must be retrained or updated when the facts change.

If a model needs access to current policies, publications, or internal
documentation, RAG is usually the better first approach.

Fine-tuning and RAG can also be combined:

```text
fine-tuning:
    teaches the desired behavior

RAG:
    supplies the current evidence
```

## 6.2 Full fine-tuning

In full fine-tuning, all model parameters are updated.

This provides maximum flexibility, but it requires substantial compute,
memory, storage, and training data. It can also make it more difficult to
preserve the model's general capabilities.

For many workshop and organizational use cases, full fine-tuning is not the
best starting point.

## 6.3 Parameter-efficient fine-tuning

Parameter-efficient fine-tuning, or PEFT, updates a relatively small number of
parameters while keeping most pretrained weights frozen.

Low-rank adaptation, or LoRA, is a widely used PEFT method.

Suppose a pretrained layer contains a weight matrix \(W\). Instead of updating
all of \(W\), LoRA learns an update represented by two smaller matrices:

```{math}
W' = W + \Delta W
```

with:

```{math}
\Delta W = BA
```

where the intermediate rank is much smaller than the dimensions of \(W\).

Conceptually:

```text
large frozen weight matrix
        +
small trainable low-rank update
        =
adapted layer
```

LoRA reduces:

- the number of trainable parameters;
- optimizer-state memory;
- the storage required for each task-specific adapter;
- the cost of experimenting with several adaptations.

It does not eliminate the need to load and execute the base model.

## 6.4 Where LoRA is applied

LoRA is commonly applied to linear projections in transformer blocks,
especially attention projections such as:

```text
query projection
key projection
value projection
attention output projection
```

Depending on the model and task, adapters may also be applied to feed-forward
projections.

There is no universal list of target-module names. Their names depend on the
model implementation.

Before configuring LoRA:

1. inspect the model architecture;
2. identify linear layers and their module names;
3. consult model-specific examples;
4. verify which parameters are trainable;
5. evaluate whether the selected targets provide sufficient capacity.

```{admonition} LoRA is still training
:class: caution

LoRA reduces training cost, but it does not remove the need for:

- clean and representative data;
- train and validation splits;
- careful hyperparameter selection;
- held-out evaluation;
- experiment tracking;
- checks for regressions and memorization.
:::

## 6.5 When fine-tuning is worth it

Fine-tuning becomes a reasonable next step when all of the following are true:

- there is a clearly defined and recurring task;
- a prompt baseline has been implemented;
- RAG has been considered when external knowledge is required;
- failures are behavioral and systematic;
- representative training examples are available;
- success can be measured on held-out data;
- the expected usage volume justifies training and maintenance;
- the organization can monitor regressions.

Warning signs that fine-tuning may be premature include:

- "We want the model to know our latest documents."
- "We have not yet created an evaluation set."
- "We do not know whether the problem is retrieval or generation."
- "We have only a handful of unrepresentative examples."
- "The prompt baseline has not been measured."
- "We want the model to stop all hallucinations."

---

# 7. Where quantization fits

Quantization represents model weights, and sometimes activations, with fewer
bits.

For example, a model may use:

```text
FP32 -> 32 bits per value
FP16 or BF16 -> 16 bits per value
INT8 -> 8 bits per value
4-bit representation -> approximately 4 bits per quantized value
```

Lower precision can reduce memory use and, on supported hardware and software,
may improve speed.

Quantization introduces a trade-off:

```text
lower memory and potentially higher throughput
                     versus
approximation error and hardware-dependent behavior
```

## 7.1 Quantization is orthogonal to the adaptation ladder

Quantization is not a fourth adaptation level.

It does not answer the same question as prompt engineering, RAG, or
fine-tuning.

```text
Prompt engineering, RAG, fine-tuning:
    How should we change model behavior or provide information?

Quantization:
    How can we represent and execute the model more efficiently?
```

A model used only for prompting can be quantized. A model in a RAG pipeline can
be quantized. A fine-tuned model can also be quantized.

## 7.2 Quantization-aware fine-tuning

In quantized LoRA training, often called QLoRA, the base model is loaded in a
low-precision quantized representation while LoRA adapters are trained using
higher-precision computation where needed.

The key idea is:

```text
quantized frozen base model
           +
trainable LoRA adapters
           =
lower-memory parameter-efficient training
```

QLoRA can make adaptation possible on hardware that cannot hold the base model
in 16-bit precision together with all required training state.

It does not make training free. Memory is still needed for:

- activations;
- LoRA parameters;
- gradients;
- optimizer state;
- temporary computation buffers.

Actual support and performance depend on the accelerator, numerical format,
kernels, framework, and model architecture.

## 7.3 Optimized training implementations

Libraries such as Unsloth aim to reduce the memory and runtime costs of
fine-tuning through optimized implementations and GPU kernels.

These tools may be useful, but they should be thought of as implementation
optimizations rather than a substitute for understanding:

- the training objective;
- data preparation;
- LoRA configuration;
- evaluation;
- hardware compatibility;
- numerical precision.

For a workshop, it is useful to first teach the standard training concepts and
then introduce optimized implementations as optional accelerators.

---

# 8. Choosing the right intervention

Use the following decision process.

## Step 1: Define the failure

Ask:

- Is the model missing instructions?
- Is it missing evidence?
- Is it following the wrong behavioral pattern?
- Is the problem actually retrieval, latency, context length, or evaluation?

## Step 2: Establish a prompt baseline

Create the simplest prompt that specifies:

- task;
- audience;
- constraints;
- output format;
- examples;
- abstention behavior.

Evaluate it on a representative test set.

## Step 3: Add retrieval if knowledge is missing

Use RAG if the answer depends on material that is:

- private;
- changing;
- too large for a fixed prompt;
- expected to be cited.

Evaluate retrieval separately from generation.

## Step 4: Fine-tune only for persistent behavior gaps

Fine-tune when the system repeatedly fails to produce the required behavior
despite adequate instructions and evidence.

Begin with PEFT unless there is a well-supported reason to update the entire
model.

## Step 5: Optimize after measuring

Apply quantization, batching, optimized kernels, or distributed training after
profiling identifies a resource or performance bottleneck.

```{admonition} Recommended order
:class: tip

1. Make it measurable.
2. Make it work with prompting.
3. Add evidence with RAG if required.
4. Fine-tune only for persistent behavioral shortcomings.
5. Optimize memory and throughput.
:::

## Decision guide

### Use prompt engineering when

- the base model can already perform the task;
- the task is easy to specify;
- required context fits in the prompt;
- iteration speed is important.

### Add RAG when

- answers depend on external or changing knowledge;
- sources must be inspectable;
- the corpus can be searched;
- missing evidence is the dominant failure.

### Fine-tune when

- the failure is systematic behavior rather than missing evidence;
- enough representative examples exist;
- the expected behavior can be evaluated;
- usage volume or quality requirements justify the lifecycle cost.

### Add quantization when

- memory prevents loading or training the selected model;
- deployment cost is a concern;
- supported low-precision kernels are available;
- measurements show that the quality and performance trade-off is acceptable.

```{exercise} Select an adaptation strategy
:label: exercise-select-strategy

For each scenario, choose one of the following starting points:

- prompt engineering;
- prompt engineering plus RAG;
- fine-tuning;
- RAG plus fine-tuning.

Explain your choice.

1. A model must answer questions about policies that change every month and
   cite the relevant policy section.
2. A model must convert thousands of recurring incident reports into a strict
   internal schema.
3. A model writes good summaries but uses the wrong headings.
4. A model must use live documentation while consistently producing commands
   in an organization's preferred style.
5. A model must answer questions about a set of papers that is updated every
   week.
```

```{solution} exercise-select-strategy
:class: dropdown

1. **Prompt engineering plus RAG.** The information changes and must be cited.
2. **Fine-tuning may be justified**, after establishing a prompt baseline.
   The problem is a recurring input-to-output behavior, assuming enough
   representative examples exist.
3. **Prompt engineering.** Explicit formatting instructions and examples are
   the least complex intervention.
4. **RAG plus fine-tuning may eventually be appropriate.** RAG supplies current
   documentation. Fine-tuning may be justified if the preferred style remains
   inconsistent after strong prompting.
5. **Prompt engineering plus RAG.** The paper collection changes frequently
   and should remain independently updateable.
```

---

# 9. Evaluation should precede complexity

Each adaptation level requires evaluation.

Without an evaluation baseline, it is impossible to know whether additional
complexity helped.

## 9.1 Separate evaluation dimensions

Useful dimensions include:

- task correctness;
- factual support;
- completeness;
- adherence to format;
- consistency;
- latency;
- memory use;
- cost;
- robustness to difficult inputs;
- appropriate abstention.

Do not collapse every dimension into one score too early. A system may become
more fluent while becoming less factually grounded.

## 9.2 Evaluate components separately

For prompt-only systems, evaluate the complete prompt on a fixed test set.

For RAG systems, separate:

```text
retrieval evaluation:
    Did we retrieve the necessary evidence?

generation evaluation:
    Did the answer use the evidence correctly?

end-to-end evaluation:
    Did the user receive a correct and useful answer?
```

For fine-tuned systems, compare:

- the unchanged base model;
- the prompt baseline;
- the fine-tuned model;
- performance on the target task;
- performance on important general capabilities.

## 9.3 Automated and human evaluation

Automated metrics can be useful for well-defined outputs. Depending on the
task, these may include:

- accuracy;
- precision and recall;
- F1 score;
- perplexity;
- task-specific text-generation metrics;
- schema-validity checks;
- retrieval recall and ranking metrics.

An LLM can also be used as an evaluator, but it should not be treated as an
objective authority. LLM-based evaluation can be sensitive to prompt wording,
model preference, output ordering, verbosity, and shared failure modes.

Human evaluation remains important when quality is contextual or subjective.

```{exercise} Design a minimal evaluation
:label: exercise-design-evaluation

Choose one of the scenarios from the previous exercise.

Design a minimal evaluation plan containing:

1. a baseline;
2. a held-out test set;
3. two quality dimensions;
4. one operational metric;
5. a criterion for deciding whether to move to the next adaptation level.
```

```{solution} exercise-design-evaluation
:class: dropdown

For the policy-question scenario, one possible plan is:

- **Baseline:** a prompt-only model without retrieval.
- **Test set:** 30 questions covering current, old, ambiguous, and
  deliberately unanswerable policy questions.
- **Quality dimension 1:** percentage of answers supported by the current
  policy.
- **Quality dimension 2:** citation correctness.
- **Operational metric:** median response latency.
- **Decision criterion:** adopt RAG if it substantially improves evidence
  support and citation correctness while keeping latency within the service
  requirement.

The precise threshold should be chosen from the application's risk and cost
requirements, not invented after seeing the results.
```

---

# 10. Summary

A language model first maps text to tokens and embeddings. Transformer blocks
then contextualize these representations.

Within a transformer block:

- attention moves information between token positions;
- queries and keys determine relevance;
- values carry the information being combined;
- causal masking prevents a decoder from using future tokens;
- feed-forward networks transform each position independently;
- residual connections and normalization support stable deep computation.

There is a rising ladder of adaptation:

```text
Prompt engineering
    simplest and cheapest
    changes instructions and context

Retrieval-augmented generation
    adds external, updateable evidence
    introduces retrieval infrastructure

Fine-tuning
    changes model parameters
    requires representative data and stronger evaluation
```

Quantization is an enabling optimization across these levels. It reduces
memory use by representing the model with fewer bits, but it does not decide
what information or behavior the application needs.

The main engineering principle is:

```text
Start with the least complex intervention that can be shown, through
evaluation, to satisfy the requirements.
```

## Key points

- Understand the model architecture before adapting it.
- Treat prompt engineering as a real baseline, not as a preliminary toy.
- Use RAG for changing, private, or citable knowledge.
- Use fine-tuning for persistent and measurable behavioral adaptation.
- Prefer parameter-efficient methods such as LoRA as an initial fine-tuning
  approach.
- Introduce quantization when memory or deployment constraints justify it.
- Evaluate before and after every increase in system complexity.

## Optional follow-up exercises

- Inspect how a tokenizer splits technical terms from your own domain.
- Visualize a causal attention mask for several sequence lengths.
- Compare three prompts on a fixed set of examples.
- Evaluate retrieval and generation failures separately in a small RAG system.
- Inspect the module names of a transformer and identify possible LoRA
  targets.
- Compare model memory use at 32-bit, 16-bit, 8-bit, and 4-bit storage.
