# What is a Large Language Model?

A Large Language Model (LLM) is a type of artificial intelligence trained on
vast amounts of text data to predict and generate human-like text. At their
core, these models learn statistical patterns in language: given a sequence of
words (or better *tokens*: fragments of words, commas, and anything in text), 
they predict what comes next.

```{figure} img/llm.png
:alt: Input tokens ->  Model -> Input tokens + output token
:width: 100%

A simplistic mental model for an LLM
```

:::{admonition} Stochastic Parrots
:class: note

{attribution="Emily M. Bender and Timnit Gebru"}
>  LM [Language Model] is a system for haphazardly stitching together sequences of linguistic forms it has observed in its vast training data, according to probabilistic information about how they combine, but without any reference to meaning: a stochastic parrot.

From: <https://doi.org/10.1145/3442188.3445922>
:::

<!---
## Features of an LLM

Without going into deep details, we can identify the key features of an LLM.

### Tokenization and embedding

LLMs (and in general all neural networks) only understand numbers. So, the first
step is to convert the input text into numbers. This is done by a process called
*tokenization*. This determines how text is split into smaller units and mapped
to integers. This again gets translated to vectors called *embedding*. We will 
shortly see why this is done.

### Attention 


--->