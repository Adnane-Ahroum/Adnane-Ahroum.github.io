---
layout: page
title: English to Darija Translation
---

<img class="project-cover" src="{{ '/projects/orange-darija/cover.svg' | relative_url }}" alt="English to Darija NMT" />

**Stack:** Python, NLP, Whisper STT, Transformer Seq2Seq  
**Repo:** [Orange-Translation-EnglishToDarija-Model](https://github.com/Adnane-Ahroum/Orange-Translation-EnglishToDarija-Model)

Neural machine translation for **English → Moroccan Darija**, developed at Orange Maroc as part of a conversational AI / call-routing stack.

> This repo is primarily notebooks (no README figures yet). Visuals above summarize the system.

## Highlights

- Whisper STT → Transformer Seq2Seq NMT pipeline
- PyDoDA corpus cleaning (30k+ entries), mixed-script normalization, BPE
- ~15% improvement in call-routing accuracy downstream

[← Back to Home]({{ '/' | relative_url }})
