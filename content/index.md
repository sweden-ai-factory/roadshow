# Fine tuning tutorial

:::{prereq}

- Basic familiarity with Python
- Some familiarity with linear algebra
- If you have used a LLM-based chat application it'd be beneficial
- Some basic understanding of how a machine learning/deep learning model is
trained at a general level
  :::

```{toctree}
:caption: The lesson
:maxdepth: 1
llm-anatomy
choosing-an-adaptation-strategy
fine-tuning
evaluating-llms
```

```{toctree}
:caption: Optional material
:maxdepth: 1
quantisation
```

```{toctree}
:caption: Reference
:maxdepth: 1

quick-reference
guide
```

## Overview

**Adapting large language models**

Large language models (LLMs) are now ubiquitous in many applications and a
variety of both open- and closed-weight are available for many disparate tasks.
However, it can be the case that an off-the-shelf model does not perform well
in a specific task, or that we would like a smaller model focused on some
specific use cases and avoid expensive, large general-purpose models. In these
cases, we need to find a way to "tweak" the behaviour of a model to fit the
requirements.
A LLM can be adapted to a task in several ways: we can change the
instructions given to the model, provide external information at inference
time, or continue training the model on examples of the desired behaviour.

These approaches form a progression:

```text
Prompt engineering -> Retrieval-augmented generation -> Fine-tuning
```

Moving from left to right gives us additional ways to influence the system. It
also introduces more infrastructure, more data preparation, more evaluation
work, and more chances for failure.

The central principle of this lesson is:

:::{admonition} Start with the simplest intervention
:class: important

Use the least complex approach that can be shown, through evaluation, to meet
the requirements of the application.
:::

Prompt engineering is therefore not merely a preliminary step before
fine-tuning. For many applications, it is the appropriate final solution.
Retrieval-augmented generation is useful when the model needs access to
external or changing information. Fine-tuning becomes useful when the
remaining problem is a persistent pattern of behaviour that instructions and
retrieved evidence do not solve adequately.

:::{figure} img/adaptation-ladder.png
:width: 95%
:alternative: Prompt engineering, retrieval-augmented generation, and fine-tuning arranged along a progression of increasing complexity and computational cost.

Prompt engineering changes the instructions. RAG adds external evidence.
Fine-tuning changes model parameters. Original figure created by VSC [here](https://gitlab.tuwien.ac.at/vsc-public/training/LLMs-on-supercomputers/-/tree/main/presentations?ref_type=heads).
:::

## Who is the lesson for?

The lesson is intended for researchers, research software engineers, and
technical staff who want to understand how large language models can be
adapted to specialist tasks.

The main focus is not on any particular model or software library (our example
will be, though). Instead, the lesson develops a conceptual framework that can
be applied when planning an LLM-based application or a fine-tuning experiment.

## Learning objectives

By the end of the lesson, learners should be able to:

- explain how text becomes a sequence of contextual token representations;
- describe the roles of tokenisation, embeddings, attention, and feed-forward
  layers;
- distinguish a problem with instructions from a problem with evidence;
- choose between prompting, RAG, and fine-tuning for a concrete use case;
- explain why fine-tuning is not a reliable substitute for document retrieval;
- describe the difference between full fine-tuning and parameter-efficient
  fine-tuning;
- explain the central idea behind low-rank adaptation;
- design a baseline and evaluation before introducing additional complexity;
- describe the main ways of evaluating a trained LLM and the limitations of
  each;
- interpret benchmark results with appropriate caution.

## Lesson schedule

A possible schedule for a three-hour lesson is:

```text
Introduction                              10 minutes
How an LLM processes text                 20 minutes
Break                                     10 minutes
Choosing an adaptation strategy           40 minutes
Fine-tuning large language models         40 minutes
Evaluating large language models          30 minutes
Discussion and conclusions                15 minutes
```

The episodes can also be taught separately. The adaptation-strategy episode
does not require a detailed mathematical understanding of attention, although
the anatomy episode provides useful context.

## Software and infrastructure

The conceptual parts of the lesson do not require a GPU. Exercises can be
completed through discussion or using an existing model interface.

A later hands-on fine-tuning lesson may use the Hugging Face ecosystem,
including Transformers, Datasets, TRL, PEFT, and Accelerate. Those
implementation details are deliberately kept separate from the conceptual
material in this lesson.

## See also

These lessons were inspired / adapted from the excellent material developed by the Vienna scientific cluster (VSC) [here](https://gitlab.tuwien.ac.at/vsc-public/training/LLMs-on-supercomputers) [License: CC BY-SA 4.0]. Moreover, the Hugging Face people published a guidebook of sorts based on their experience training SmolLM, which can be found [here](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#beyond-base-modelspost-training-in-2025).

::::{admonition} License
:class: attention

:::{admonition} CC BY-SA for media and pedagogical material
:class: attention dropdown

Copyright © 2026 Sweden AI Factory. This material is released by Sweden AI Factory under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

**Canonical URL**: <https://creativecommons.org/licenses/by-sa/4.0/>

[See the legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)

## You are free to

1. **Share** — copy and redistribute the material in any medium or format for any purpose, even commercially.
2. **Adapt** — remix, transform, and build upon the material for any purpose, even commercially.
3. The licensor cannot revoke these freedoms as long as you follow the license terms.

## Under the following terms

1. **Attribution** — You must give [appropriate credit](https://creativecommons.org/licenses/by-sa/4.0/#ref-appropriate-credit) , provide a link to the license, and [indicate if changes were made](https://creativecommons.org/licenses/by-sa/4.0/#ref-indicate-changes) . You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
2. **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the [same license](https://creativecommons.org/licenses/by-sa/4.0/#ref-same-license) as the original.
3. **No additional restrictions** — You may not apply legal terms or [technological measures](https://creativecommons.org/licenses/by-sa/4.0/#ref-technological-measures) that legally restrict others from doing anything the license permits.

## Notices

You do not have to comply with the license for elements of the material in the public domain or where your use is permitted by an applicable [exception or limitation](https://creativecommons.org/licenses/by/4.0/deed.en#ref-exception-or-limitation) .

No warranties are given. The license may not give you all of the permissions necessary for your intended use. For example, other rights such as [publicity, privacy, or moral rights](https://creativecommons.org/licenses/by/4.0/deed.en#ref-publicity-privacy-or-moral-rights) may limit how you use the material.

This deed highlights only some of the key features and terms of the actual license. It is not a license and has no legal value. You should carefully review all of the terms and conditions of the actual license before using the licensed material.

:::

:::{admonition} MIT for source code and code snippets
:class: attention dropdown

MIT License

Copyright (c) 2026, Sweden AI Factory project, {{ author }}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

:::

::::
