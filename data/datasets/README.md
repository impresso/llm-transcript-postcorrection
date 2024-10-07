# Data Description

The folder presents the eight post-correction benchmarks, each comprising two historical transcriptions: an automated system's output needing correction, and its corresponding ground truth. These versions are aligned at various levels without any images of the original documents. The selection of benchmarks was influenced by diversity and the need for transcripts of adequate length to offer sufficient context for Large Language Models (LLMs).

| Dataset       | Document Type            | Origin | Time Period | Language               | # Lines | # Sentences | # Regions |
|---------------|--------------------------|--------|-------------|------------------------|---------|-------------|-----------|
| icdar-2017    | newspapers, monographies | OCR    | 17C-20C     | en, fr                 | 0       | 461         | 28        |
| icdar-2019    |                          | OCR    | not specified | bg, cz, en, fr, de, pl, sl | 0   | 404         | 41        |
| overproof     | newspaper                | OCR    | 19-20C      | en                     | 2,278   | 399         | 41        |
| impresso-nzz  | newspaper                | OCR    | 18-20C      | de                     | 1,256   | 577         | 203       |
| ajmc-mixed    | class. commentaries      | OCR    | 19C         | grc, de, en, fr        | 535     | 379         | 33        |
| ajmc-primary  | class. commentaries      | OCR    | 19C         | grc, de, en, fr        | 40      | 27          | 9         |
| htrec         | papyri and manuscripts   | HTR    | 10C-16C     | grc                    | 180     | 8           | 8         |
| ina           | radio programs           | ASR    | 20C         | fr                     | 201     | 290         | 6         |

### Data Copyrights

The datasets are publicly available and have been collected from various sources. The original datasets are available under different licenses, and the data in this repository is provided under the same licenses as the original datasets.
* icdar-2017: [ICDAR2017 Competition on Post-OCR Text Correction](https://hal.science/hal-03025499v1/file/icdar2017-competition-post%283%29.pdf)
* icdar-2019: [ICDAR2019 Competition on Post-OCR Text Correction](https://sites.google.com/view/icdar2019-postcorrectionocr)
* overproof: [Overproof OCR Dataset](https://overproof.projectcomputing.com/evaluation)
* impresso-nzz: internal dataset available only for research purposes, copyright type: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
* ajmc-mixed: internal dataset available only for research purposes, copyright type: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
* ajmc-primary: internal dataset available only for research purposes, copyright type: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
* htrec: [HTRec Handwritten Text Recognition Dataset](https://www.aicrowd.com/challenges/htrec-2022)
* ina: internal dataset available only for research purposes, copyright type: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
