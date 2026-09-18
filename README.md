# Python Literacy for Text-as-Data

A 3-day hands-on workshop taking social scientists with no programming background from their first
line of Python to a fine-tuned classifier they trained themselves. Everything runs in free
browser-based notebooks.

## Before Day 1

- [`prereq.ipynb`](./prereq.ipynb). Twenty minutes, self-paced. Gets Colab working, walks you
  through one deliberate error, and downloads the small embedding model Day 1 needs. You need no
  Manifesto Project account yet, we set that up together at the end of Day 1.
- [`prework/encoder_cheatsheet.html`](./prework/encoder_cheatsheet.html). One page on what a
  BERT-style encoder is, in plain language. No code. Print it from
  [`encoder_cheatsheet.pdf`](./prework/encoder_cheatsheet.pdf).
- [`prework/rosetta.html`](./prework/rosetta.html). The R to Python map. Keep it open all three
  days, or print [`rosetta.pdf`](./prework/rosetta.pdf).

## Day 1. Python basics through a real chore

Files in [`day1/`](./day1/).

- `session1_slides.html` and `session2_slides.html`. Self-contained decks. Arrow keys to move, `F`
  for fullscreen.
- `day1_python_basics.ipynb`. Lists, dictionaries, comprehensions, and a first pandas table.
- `day1_record_linkage.ipynb`. Two country-name datasets that do not line up, merged three ways.
  A naive join gets 2 of 43, `difflib` gets 29, and sentence embeddings get 38.
- `countries_worldbank.csv`, `countries_source.csv`, `toy_manifesto.csv`. The shipped data. No key
  and no GPU needed.
- `glossary.html` and `glossary.pdf`. Day 1 terms.

## Day 2. Classic text-as-data, then the encoder

Files in [`day2/`](./day2/).

- `session3_slides.html` and `session4_slides.html`.
- `day2_topics.ipynb`. Pull the corpus, build a document-term matrix, fit an LDA topic model, then
  meet the Structural Topic Model by calling R's `stm` from Python through rpy2.
- `day2_inputs.ipynb`. What a tokenizer does, what `input_ids` and `attention_mask` are, and how to
  assemble every input a trainer needs, stopping just short of training.
- `glossary.html` and `glossary.pdf`. Topic model and tokenization terms.

## Day 3. Fine-tune, evaluate honestly, ship

Files in [`day3/`](./day3/).

- `session5_slides.html` and `session6_slides.html`.
- `day3_finetune.ipynb`. The capstone. Train a left-right classifier with SetFit on a free T4,
  score it against the TF-IDF baseline you built on Day 2, apply it across a corpus, and save it.
- `day3_lora_variant.ipynb`. The stretch notebook. The same job with the Hugging Face Trainer and a
  LoRA adapter.
- `reproducibility_checklist.html` and its PDF. What to record before you write up a
  fine-tuned measure.

Every one-pager ships as HTML and as a single-page A4 PDF, so you can read it on screen or print it.

## The helper module

[`workshop_utils.py`](./workshop_utils.py) holds the three functions the notebooks share. Each
notebook fetches it with a one-line `wget`, so there is nothing to install.

- `pull_manifesto(api_key)`. Downloads annotated quasi-sentences with their party, year and country.
- `collapse_rile(code)`. Folds a Manifesto category code into `left`, `right` or `None`.
- `manifesto_stopwords(df)`. English stopwords plus the country and party names in your corpus,
  which otherwise win topics of their own.

Run `python workshop_utils.py` to execute its self-check.

## Data

The corpus is the [Manifesto Project](https://manifesto-project.wzb.eu) Corpus, main dataset version
`MPDS2026a` and corpus version `2026-1`. Their terms forbid redistribution, so this repository ships
code and every participant pulls their own copy with their own free API key. Registration is free
and instant.

Please cite the Manifesto Project if you use the data in published work, and read their
[terms of use](https://manifesto-project.wzb.eu/information/documents/terms_of_use).

## Models

- `sentence-transformers/all-MiniLM-L6-v2`. Apache 2.0. Used for semantic record linkage on Day 1
  and as the SetFit backbone on Day 3.
- `distilbert-base-uncased`. Apache 2.0. Used on Day 2 to show tokenization and a forward pass.
