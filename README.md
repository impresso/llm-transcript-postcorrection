[![DOI](https://zenodo.org/badge/DOI/10.5281/xyz)](https://doi.org/10.5281/xyz)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python)](https://www.python.org/) 
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-EE4C2C.svg?logo=pytorch)](https://pytorch.org/docs/2.0/) 
[![MIT](https://img.shields.io/badge/License-MIT-3DA639.svg?logo=open-source-initiative)](LICENSE)

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

### Citation

```
@article{boros_post_2024,
  title={Post-correction of Historical Text Transcripts with Large Language Models: An Exploratory Study},
  author={Boros, Emanuela and Ehrmann, Maud and Romanello, Matteo and Najem-Meyer, Sven and Kaplan, Fr{\'e}d{\'e}ric},
  journal={LaTeCH-CLfL 2024},
  pages={133--159},
  year={2024},
  publisher={The Association for Computational Linguistics}
}
```
