---
title: Choosing an adaptation strategy
teaching: 45
exercises: 20
---

# Choosing an adaptation strategy - aka when NOT to fine-tune

When an LLM application performs poorly, fine-tuning is rarely the only
possible response. The model may need clearer instructions, better context,
access to external information, or a more suitable evaluation procedure.

Prompt engineering, retrieval-augmented generation, and fine-tuning address
different sources of failure:

```text
Prompt engineering:
    change the instructions and examples

Retrieval-augmented generation:
    supply external evidence at inference time

Fine-tuning:
    change model parameters through training
```

These methods can be combined, but they should not be introduced
indiscriminately. Every additional component has to be developed, evaluated,
operated, and maintained.

:::{questions}

- What can prompt engineering change?
- When does a model need retrieval rather than better instructions?
- What kinds of behaviour justify fine-tuning?
- Why is fine-tuning a poor substitute for a document store?
- How can we evaluate whether additional complexity is worthwhile?
- When should we decide not to fine-tune?

:::

:::{objectives}

By the end of this episode, learners should be able to:

- distinguish a problem with instructions from a problem with evidence;
- explain what prompt engineering, RAG, and fine-tuning change;
- recognize applications for which fine-tuning is unnecessary or
  inappropriate;
- choose an initial adaptation strategy for a concrete use case;
- define a baseline against which a more complex system can be evaluated;
- separate retrieval failures from generation failures;
- describe when RAG and fine-tuning may usefully be combined.

:::

## Three ways to influence a model

The progression from prompting to RAG to fine-tuning represents increasing
system complexity.

```{figure} figures/adaptation-ladder.png
:alt: Prompt engineering, retrieval-augmented generation, and fine-tuning
      arranged along a progression of increasing complexity and computational
      cost.
:width: 95%
:class: img-responsive

Prompt engineering changes the instructions. RAG adds external evidence.
Fine-tuning changes model parameters.
```

The progression should not be interpreted as a sequence that every project
must complete. A prompt-only system is not an unfinished RAG system, and a RAG
system is not an unfinished fine-tuning project.

The objective is to find the least complex system that meets the application's
requirements.

:::{admonition} The central diagnostic question
:class: important

Is the model failing because it does not know what to do, because it lacks the
necessary evidence, or because it cannot reliably perform the required
behaviour?
:::

Those three cases point toward prompting, retrieval, and fine-tuning
respectively. Real systems may contain more than one kind of failure, but the
distinction provides a useful starting point.

## Prompt engineering

Prompt engineering changes the model's input. It does not update the model
parameters.

A prompt can specify the task, the intended reader, the required structure,
the available evidence, and what the model should do when it cannot answer.
It can also contain examples of suitable inputs and outputs.

Consider this prompt:

```text
Summarize this paper.
```

A more useful version might be:

```text
Summarize the supplied paper for research software engineers who understand
machine learning but are not specialists in this application domain.

Use the following headings:

- Research question
- Method
- Main result
- Limitations
- Reproducibility

Use at most 300 words. Do not invent information that is absent from the
paper. If reproducibility information is not reported, write "not reported in
the supplied text".
```

The second prompt reduces ambiguity. It defines the audience, structure,
length, and handling of missing information.

### Direction, format, and examples

A useful prompt normally establishes what the model is supposed to do before
it attempts to control tone or style.

The following elements are often enough to create a strong baseline:

```text
Task:
What should the model do?

Context:
What information may it use?

Constraints:
What must it avoid?

Output:
What structure should it return?

Examples:
What does a good input-output pair look like?

Abstention:
What should happen when the evidence is insufficient?
```

Examples are particularly helpful when a desired format is difficult to
describe precisely. If the task is to classify technical incidents, a few
representative examples can establish distinctions that would otherwise
require a long definition.

### Breaking a task into stages

A complicated task can sometimes be made more reliable by separating it into
steps:

```text
source text
    -> extract claims
    -> identify supporting evidence
    -> mark unsupported claims
    -> produce a revised response
```

The steps may appear in a single structured prompt or in separate model calls.
Separate calls make intermediate results easier to inspect, although they also
increase latency and introduce additional failure points.

### When prompting is enough

Prompt engineering is often sufficient when the base model already has the
required capability and all necessary information can be included in the
context.

Examples include rewriting text for a known audience, extracting fields from
a short document, producing a specified format, explaining supplied material,
or generating a first draft that will be reviewed by a person.

If a model produces good content under the wrong headings, the first response
should be to specify the headings. Training a model to solve a problem that can
be expressed in two lines of instructions is difficult to justify.

:::{admonition} Do not confuse variation with incapability
:class: note

A generative model may produce different wording across repeated calls. If the
application requires a rigid structure, first try explicit output constraints,
examples, schema validation, and deterministic or low-temperature decoding.

Fine-tuning may improve consistency, but it should not be the first mechanism
used to impose a simple format.
:::

## Retrieval-augmented generation

Retrieval-augmented generation (RAG) adds an external
information-retrieval stage.

A simplified RAG system looks like this:

```text
user question
    -> retrieval query
    -> relevant passages
    -> prompt containing question and passages
    -> generated answer
```

The model parameters are not normally changed. The model receives additional
evidence as part of its input.

### What problem does RAG solve?

RAG is appropriate when the model cannot answer reliably from its existing
parameters and the required information exists in an external collection.

Typical collections include internal documentation, scientific papers,
technical manuals, organizational policies, support records, and databases
whose contents change over time.

Retrieval is also useful when an answer must be accompanied by inspectable
sources. A model may have encountered a fact during pretraining, but its
parameters do not provide a dependable citation mechanism. A RAG system can
retain the identifiers of the retrieved passages and ask the model to cite
them.

A basic grounded-generation prompt could be:

```text
Answer the question using only the supplied sources.

If the sources do not contain enough information, state that the available
evidence is insufficient.

Cite the source identifier after every factual claim.

Question:
{question}

Sources:
{retrieved_passages}
```

This prompt does not by itself create a reliable RAG system. The documents
must be indexed, the query must retrieve useful passages, and the selected
context must preserve the information needed to answer the question.

### Retrieval and generation are separate problems

RAG systems can fail before the model begins to generate.

The relevant document may be absent from the index. The query may use
different terminology from the source. A relevant paragraph may have been
split across chunk boundaries. Metadata filters may exclude the correct
document. A ranking system may place old or irrelevant material above the
current source.

Generation can fail even when retrieval succeeds. The model may ignore the
relevant passage, combine incompatible sources, choose an outdated statement,
or produce a claim that is not supported by the retrieved material.

These failures require different remedies. Fine-tuning the generator will not
repair a missing document or a faulty metadata filter.

### When RAG is worth introducing

RAG is a strong candidate when the application depends on information that is
private, frequently updated, too large to include in every prompt, or expected
to be cited.

It is less useful when the problem contains no substantial knowledge-retrieval
component. If the task is to convert a known input into a fixed schema, adding
a vector database may introduce infrastructure without addressing the actual
failure.

:::{admonition} RAG does not eliminate hallucinations
:class: caution

Retrieval can ground the response in external evidence, but the model can
still misread, ignore, or go beyond that evidence.

A grounded system needs explicit instructions, suitable retrieval, source
tracking, and an evaluation that checks whether claims are supported.
:::

## Fine-tuning

Fine-tuning continues training a pretrained model on examples chosen for a
particular objective. It changes some or all of the model parameters.

This makes fine-tuning fundamentally different from prompting and retrieval.
A prompt or retrieved document can be replaced immediately. A parameter
update is produced through a training process and is less directly
inspectable.

Fine-tuning is most useful when the remaining problem is a repeated,
measurable pattern of behaviour. Examples include using a specialized
output representation, mapping recurring inputs to recurring outputs, or
following domain conventions that are cumbersome to demonstrate in every
prompt.

Fine-tuning may also teach a model to use specialist vocabulary more
consistently. It is not, however, a dependable mechanism for loading an
organization's changing documents into the model.

## When not to fine-tune

Fine-tuning is often proposed too early. Before training, it is worth checking
whether the proposed use case matches any of the following situations.

### The model needs current information

If the answer depends on policies, publications, prices, software versions, or
other information that changes, the system needs access to a current source.

Training examples are a poor update mechanism for such material. Updating the
source collection is faster and easier to inspect than training another
adapter. RAG is usually the better starting point.

### The answer must be traceable to a source

A model parameter cannot provide a reliable record of which document supports
an answer. Even if a training example contained the relevant statement, the
generated response may combine it with other learned associations.

If users need citations or evidence, the evidence should be retrieved and
passed to the model explicitly.

### The problem is a vague prompt

If the original instruction is:

```text
Analyze this.
```

the model has not been told what kind of analysis is required. Training should
not be used as a substitute for defining the task.

First specify the questions to answer, the audience, the output structure, and
the criteria for a satisfactory result.

### The problem is a simple output constraint

A model that uses the wrong headings or returns prose instead of JSON may need
clearer formatting instructions, examples, constrained decoding, or schema
validation.

Fine-tuning may eventually improve format adherence at scale, but a prompt
baseline should be attempted and measured first.

### There is no evaluation set

Without a held-out evaluation set, there is no reliable way to tell whether
fine-tuning improved the application.

Training loss is not sufficient. A model may fit the training examples while
failing on new inputs. It may also improve on the target task while losing
useful general behaviour.

If success has not been defined before training, the project is not ready for
fine-tuning.

### There are too few representative examples

A small set of carefully selected examples can be useful, but examples that
cover only easy or repetitive cases may teach an overly narrow pattern.

Creating more examples is not enough if those examples are generated from the
same template and contain the same blind spots. The data must represent the
variation expected in real use.

### The failure comes from retrieval

If the relevant passage is not in the model's context, training the generator
does not directly solve the missing-evidence problem.

Retrieve and inspect the evidence first. Only after retrieval is working
should the project ask whether the generator still behaves inadequately.

### The objective is to eliminate all hallucinations

Fine-tuning cannot guarantee that a generative model will never produce an
unsupported statement.

Risk must instead be managed through system design. Depending on the
application, this may include retrieval, restricted outputs, validation,
abstention, citations, human review, and limiting the model's authority.

### The application is still exploratory

Training adds a model artifact, training code, data versions, hyperparameters,
evaluation, and deployment decisions. If the task is changing every few days,
those artifacts can become obsolete before they are properly evaluated.

Prompting is usually a faster way to explore the task and collect evidence
about recurring failures.

:::{admonition} A useful rule
:class: important

Do not fine-tune to discover what the application should do.

Define the task through prompting and evaluation first. Fine-tune when the
task is stable and the remaining shortcomings are repeated enough to learn
from examples.
:::

## A practical decision process

The adaptation decision begins with a failure analysis rather than a choice of
technology.

### Establish the task and evaluation

Write down what a satisfactory output looks like. Assemble examples that
cover ordinary cases, difficult cases, ambiguous cases, and cases where the
system should decline to answer.

The evaluation need not initially be large. It does need to be representative
enough to expose whether a change solves the intended problem.

### Build a prompt baseline

Create the simplest prompt that specifies the task, context, constraints,
output, and abstention behaviour. Include examples where they clarify the
desired mapping.

Run the evaluation set and record the results. This baseline provides evidence
against which retrieval or fine-tuning can be compared.

### Add retrieval when evidence is missing

Inspect failed examples. If the model could answer correctly when given the
right document passage, the problem points toward retrieval.

Evaluate whether the retrieval stage finds that passage. Do not evaluate only
the final prose.

### Consider fine-tuning for persistent behaviour

Fine-tuning becomes reasonable when the system has appropriate instructions
and evidence but still fails in a repeated, measurable way.

At that point, the training examples can target a stable behaviour rather than
compensating for an undefined prompt or missing source.

The resulting process is:

```text
define success
    -> build a prompt baseline
    -> inspect failures
    -> add retrieval if evidence is missing
    -> inspect failures again
    -> fine-tune if behaviour remains inadequate
```

## Combining RAG and fine-tuning

RAG and fine-tuning are not mutually exclusive.

A useful division of labour is:

```text
RAG:
    supplies the current evidence

Fine-tuning:
    teaches the model how to use that evidence
```

For example, a technical assistant may need retrieval to access the current
software documentation. It may also benefit from fine-tuning if it
systematically fails to express commands according to an organization's
conventions.

The combined system is more complex than either component alone. Its
evaluation must determine whether a failure came from retrieval, prompting,
the fine-tuned behaviour, or the interaction between them.

## Evaluating the adaptation ladder

Evaluation should precede an increase in complexity. Otherwise, there is no
reliable way to tell whether the additional component helped.

### Evaluate more than fluency

A response can sound better without becoming more correct. Evaluation should
therefore separate dimensions that matter to the application.

For a document question-answering system, useful dimensions might include
factual support, citation correctness, completeness, appropriate abstention,
latency, and cost.

For a structured extraction task, useful dimensions might include field
accuracy, schema validity, handling of missing values, and robustness to
unusual inputs.

### Evaluate RAG in parts

A RAG system has at least three evaluation targets:

```text
Retrieval:
Did the system retrieve the evidence required to answer?

Generation:
Did the model use the retrieved evidence correctly?

End-to-end:
Did the user receive a satisfactory answer?
```

A good final answer does not prove that retrieval is robust. The model may
already know the answer or may have guessed correctly. Conversely, a poor
final answer does not prove that retrieval failed. The correct source may have
been present but used incorrectly.

### Evaluate fine-tuning against meaningful baselines

A fine-tuned model should be compared with the unchanged base model using the
best prompt developed during exploration.

Depending on the application, the comparison may also include:

```text
base model with a simple prompt
base model with the strongest prompt
base model with prompt and retrieval
fine-tuned model with the same retrieval
```

This prevents a training run from receiving credit for improvements that came
from a better prompt, a changed dataset, or a different retrieval setup.

Held-out evaluation is essential. Examples used during training cannot show
whether the model generalizes to new inputs.

### Automated and human evaluation

Automated checks work well when correctness can be defined precisely. Schema
validation, exact field comparison, classification scores, citation matching,
and retrieval recall are useful examples.

Human evaluation is needed when quality depends on context, usefulness, or
specialist judgment.

Another language model can assist with evaluation, but its judgment is not
objective. LLM-based evaluators can respond to superficial features such as
verbosity, ordering, or writing style. Their results should be calibrated
against human decisions and combined with deterministic checks where
possible.

## Exercise: Choose the first intervention

:::{exercise} Choose an adaptation strategy
:label: exercise-adaptation-strategy

For each scenario, decide what you would try first:

- prompt engineering;
- prompt engineering with RAG;
- fine-tuning after establishing a prompt baseline;
- RAG followed, if necessary, by fine-tuning.

Explain what kind of failure you expect the chosen approach to address.

1. A model must answer questions about policies that change every month and
   cite the relevant section.
2. A model writes useful summaries but uses the wrong headings.
3. A model must convert thousands of recurring incident reports into a strict
   internal representation.
4. A support assistant must use the current documentation and express commands
   according to an organization's conventions.
5. A model must answer questions about a collection of papers that is updated
   every week.
:::

:::{solution} exercise-adaptation-strategy
:class: dropdown

For the monthly policies, start with prompt engineering and RAG. The dominant
problem is access to changing, citable evidence.

For the incorrect headings, start with prompt engineering. The desired change
is simple to specify and should not initially require training.

For recurring incident reports, establish a prompt baseline first.
Fine-tuning may become worthwhile if the mapping is stable, the volume is
high, representative examples are available, and prompt-based output remains
insufficiently reliable.

For the support assistant, use RAG to provide current documentation. If a
strong prompt does not produce the required command style consistently,
fine-tuning may then be justified for the behavioural part of the task.

For the changing paper collection, start with prompt engineering and RAG. The
collection should be updateable independently of the model parameters.
:::

## Exercise: Decide whether to fine-tune

:::{exercise} Fine-tuning readiness review
:label: exercise-finetuning-readiness

A team proposes fine-tuning a model on 200 internal question-answer pairs.

The stated goal is:

> We want the model to know our documentation and stop hallucinating.

Before approving the training work, write at least five questions that the
team should answer.

Identify which answers would suggest that fine-tuning is premature.
:::

:::{solution} exercise-finetuning-readiness
:class: dropdown

Useful questions include:

1. How often does the documentation change?
2. Must answers cite a document or section?
3. Has the team evaluated a prompt-only baseline?
4. Has the team evaluated a retrieval-based baseline?
5. Are the 200 examples representative of real questions?
6. Is there a held-out evaluation set?
7. What does "stop hallucinating" mean in measurable terms?
8. Do failures occur because evidence is missing, or because the model uses
   available evidence incorrectly?
9. What should the system do when the documentation does not contain an
   answer?
10. Who will maintain and evaluate the model when documentation changes?

Fine-tuning is premature if the documentation changes regularly, citations
are required, retrieval has not been tested, the same examples will be used
for training and evaluation, or success has not been defined.

The phrase "know our documentation" points primarily toward retrieval. The
team may later fine-tune how the model responds to retrieved material, but
retrieval and abstention should be addressed first.
:::

## Exercise: Design an evaluation

:::{exercise} Design a comparison
:label: exercise-evaluation-comparison

Choose one scenario from the previous exercise.

Define:

1. a prompt baseline;
2. a held-out test set;
3. two quality measures;
4. one operational measure;
5. a criterion for introducing the next level of complexity.
:::

:::{solution} exercise-evaluation-comparison
:class: dropdown

For the policy-question scenario, a minimal plan could be:

**Prompt baseline:** Ask the unchanged model to answer using policy text
included directly in the prompt. Require citations and abstention when the
text is insufficient.

**Test set:** Use current, outdated, ambiguous, and unanswerable policy
questions. Keep the test questions separate from prompt examples and any
future training data.

**Quality measures:** Measure whether each claim is supported by the supplied
policy and whether citations identify the correct section.

**Operational measure:** Record the median end-to-end response time.

**Decision criterion:** Introduce retrieval if the model answers reliably when
the correct passage is supplied but the full policy collection cannot be
placed in every prompt. Consider fine-tuning only if retrieval supplies the
correct evidence and a strong prompt still produces a repeated behavioural
failure.
:::

## Summary

Prompt engineering, RAG, and fine-tuning change different parts of an LLM
application.

Prompt engineering changes the instructions and examples. It is the correct
starting point when the model already has the necessary capability and
evidence.

RAG retrieves external information and includes it in the model's context. It
is the usual choice for private, changing, extensive, or citable knowledge.

Fine-tuning changes model parameters. It is best suited to stable and repeated
behavioural patterns that remain inadequate after a strong prompt baseline
and, where necessary, reliable retrieval.

The progression is therefore not:

```text
start simple -> eventually fine-tune
```

It is:

```text
start simple
    -> evaluate
    -> diagnose the remaining failure
    -> introduce only the component that addresses that failure
```

Fine-tuning is usually the wrong response when the model needs current
documents, citations, clearer instructions, a simple output constraint, or a
working retrieval system. It is also premature when there is no representative
training data or held-out evaluation set.
