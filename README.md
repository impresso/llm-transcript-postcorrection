# Post-correction of Historical Text Transcripts with Large Language Models
*An Exploratory Study (2023)*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python)](https://www.python.org/) 
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-EE4C2C.svg?logo=pytorch)](https://pytorch.org/docs/2.0/) 
[![MIT](https://img.shields.io/badge/License-MIT-3DA639.svg?logo=open-source-initiative)](LICENSE)

- [Overview](#overview)
- [Repository Organisation](#repository-organisation)
- [Requirements](#requirements)
- [Models](#models)
- [Datasets](#datasets)
- [Experiments](#experiments)
- [Publication](#publication)
- [Follow-up Work: ICDAR 2026 HIPE-OCRepair Competition](#follow-up-work-icdar-2026-hipe-ocrepair-competition)
- [About](#about)

## Overview

The quality of transcriptions of heritage documents produced by optical character recognition (OCR, for printed documents), handwritten text recognition (HTR, for manuscripts) or automatic speech recognition  (ASR, for audio documents) has a major impact on the ability to search and process historical texts. This is the implementation of [Post-correction of Historical Text Transcripts with Large Language Models: An Exploratory Study](https://infoscience.epfl.ch/record/307961).

#### Can LLMs amend poor historical transcriptions?

* **Ability to correct.** Do LLMs improve, degrade, or leave the input text intact?
* **Sensitivity to variations in input text and instructions.** Does LLM post-correction performance depend on the noise of the original document? How sensitive is it to prompt instructions?
* **Real-world applicability.** How do open-access models compare with the limited-access ones? Could millions of noisy historical documents be easily corrected?

## Repository organisation

```
llm-transcript-postcorrection/
├── data/
│   ├── datasets/
│   │   └── ina/*txt,*xml
│   └── prompts/
│   └── config.yml
├── lib/
│   ├── converters/
│   └── modules/
└── notebooks/
    └── ResultsGeneration.ipynb
```

* `lib`: main codebase for the experiments. Details [here](lib/).
* `notebooks`: Jupyter notebooks for data error analysis. Details [here](notebooks/).
* `data`: the data samples utilised in our experimental setup. Details [here](data/).

### Requirements
`>= python 3.9`

```bash
# install dependencies
pip install -r requirements.txt
# install transformers from the repository
pip install git+https://github.com/huggingface/transformers
```

### Models

Details [here](lib/).

| Model     | Release Date | Sizes         | Access  | Max Length |
|-----------|--------------|---------------|---------|------------|
| GPT-2     | 11.2019      | 1.5B          | Open    | 1,024      |
| GPT-3     | 06.2020      | 175B          | Limited | 2,049      |
| GPT-3.5   | 03.2023      | Unknown       | Limited | 4,096      |
| GPT-4     | 03.2023      | Unknown       | Limited | 8,192      |
| BLOOM     | 07.2022      | 560M, 3B, 7.1B | Open    | 2,048      |
| BLOOMZ    | 11.2022      | 560M, 3B, 7.1B | Open    | 2,048      |
| OPT       | 05.2022      | 350M, 6.7B    | Open    | 2,048      |
| LLaMA     | 02.2023      | 7B            | Open    | 2,048      |
| LLaMA-2   | 07.2023      | 7B            | Open    | 4,096      |

### Datasets

Details [here](data/).

| Dataset       | Document Type            | Origin | Time Period | Language               | # Lines | # Sentences | # Regions |
|---------------|--------------------------|--------|-------------|------------------------|---------|-------------|-----------|
| icdar-2017    | newspapers, monographies | OCR    | 17C-20C     | en, fr                 | 0       | 461         | 28        |
| icdar-2019    | OCR                      |        | not specified | bg, cz, en, fr, de, pl, sl | 0   | 404         | 41        |
| overproof     | newspaper                | OCR    | 19-20C      | en                     | 2,278   | 399         | 41        |
| impresso-nzz  | newspaper                | OCR    | 18-20C      | de                     | 1,256   | 577         | 203       |
| ajmc-mixed    | class. commentaries      | OCR    | 19C         | grc, de, en, fr        | 535     | 379         | 33        |
| ajmc-primary  | class. commentaries      | OCR    | 19C         | grc, de, en, fr        | 40      | 27          | 9         |
| htrec         | papyri and manuscripts   | HTR    | 10C-16C     | grc                    | 180     | 8           | 8         |
| ina           | radio programs           | ASR    | 20C         | fr                     | 201     | 290         | 6         |

### Experiments

Details [here](lib/).

### Publication

Emanuela Boros, Maud Ehrmann, Matteo Romanello, Sven Najem-Meyer, and Frédéric Kaplan. 2024. [Post-Correction of Historical Text Transcripts with Large Language Models: An Exploratory Study](https://aclanthology.org/2024.latechclfl-1.14/). In Proceedings of the 8th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2024), March 2024. Association for Computational Linguistics, St. Julians, Malta, 133–159. https://doi.org/10.18653/v1/2024.latechclfl-1.14

```bibtex
@inproceedings{boros_postcorrection_2024,
  title = {Post-{{Correction}} of {{Historical Text Transcripts}} with {{Large Language Models}}: {{An Exploratory Study}}},
  booktitle = {Proceedings of the 8th {{Joint SIGHUM Workshop}} on {{Computational Linguistics}} for {{Cultural Heritage}}, {{Social Sciences}}, {{Humanities}} and {{Literature}} ({{LaTeCH-CLfL}} 2024)},
  author = {Boros, Emanuela and Ehrmann, Maud and Romanello, Matteo and {Najem-Meyer}, Sven and Kaplan, Fr{\'e}d{\'e}ric},
  editor = {Bizzoni, Yuri and {Degaetano-Ortlieb}, Stefania and Kazantseva, Anna and Szpakowicz, Stan},
  year = 2024,
  month = mar,
  pages = {133--159},
  publisher = {Association for Computational Linguistics},
  address = {St. Julians, Malta},
  url = {https://aclanthology.org/2024.latechclfl-1.14/},
  urldate = {2026-07-09}
}
```
## Follow-up work: ICDAR 2026 HIPE-OCRepair Competition

- [HIPE-OCRepair 2026 website](https://hipe-eval.github.io/HIPE-OCRepair-2026/)
- [HIPE-eval shared task series website](https://hipe-eval.github.io/)
- [HIPE-OCRepair data](https://github.com/hipe-eval/HIPE-OCRepair-2026-data)
- [HIPE-OCRepair evaluation toolkit](https://github.com/hipe-eval/HIPE-OCRepair-2026-eval)

## About

### Impresso project

[Impresso - Media Monitoring of the Past](https://impresso-project.ch) is an interdisciplinary research project that aims to develop and consolidate tools for processing and exploring large collections of media archives across modalities, time, languages and national borders. The first project (2017-2021) was funded by the Swiss National Science Foundation under grant No. [CRSII5_173719](http://p3.snf.ch/project-173719) and the second project (2023-2027) by the SNSF under grant No. [CRSII5_213585](https://data.snf.ch/grants/grant/213585) and the Luxembourg National Research Fund under grant No. 17498891.

### Copyright

Copyright (C) 2023 The Impresso team.

### License

This program is provided as open source under the [GNU Affero General Public License](https://github.com/impresso/impresso-pyindexation/blob/master/LICENSE) v3 or later.

---

<p align="center">
  <img src="https://github.com/impresso/impresso.github.io/blob/master/assets/images/3x1--Yellow-Impresso-Black-on-White--transparent.png?raw=true" width="350" alt="Impresso Project Logo"/>
</p>

