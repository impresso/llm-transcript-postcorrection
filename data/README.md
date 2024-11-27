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
---
### Organisation

````
data/
├── datasets/
│   ├── asr/
│   │   ├── original/
│   │   │   └── ina/
│   │   │       ├── *txt
│   │   │       └── *xml
│   │   └── converted/
│   │       └── ina.jsonl
│   ├── htr/
│   │   ├── original/
│   │   │   └── htrec/
│   │   │       └── *csv
│   │   └── converted/
│   │       └── htrec.jsonl
│   └── ocr/
│       ├── original/
│       │   ├── ajmc/
│       │   │   └── *tsv
│       │   ├── icdar-2017/
│       │   │   └── *txt
│       │   ├── icdar-2019/
│       │   │   └── *txt
│       │   ├── impresso-nzz/
│       │   │   └── *xml
│       │   └── overproof/
│       │       └── *txt
│       └── converted/
│           ├── ajmc/
│           │   └──  ajmc.jsonl
│           ├── icdar-2017/
│           │   └── icdar-2017.jsonl
│           ├── icdar-2019/
│           │   └── icdar-2019.jsonl
│           ├── impresso-nzz/
│           │   └── impresso-nzz.jsonl
│           └── overproof/
│               └── overproof.jsonl
└── prompts/
    ├── prompt1.txt
    ├── prompt2.txt
    ├── prompt3.txt
    ...
    └── promptN.txt
````

- `datasets/`: Root directory containing all data related to the project.
  - `asr/`: Directory for Automatic Speech Recognition (ASR) data.
    - `original/`: Contains the original ASR data files.
      - `ina/*txt, *xml`: Original ASR files for the INA dataset, in text and XML formats.
    - `converted/`: Contains converted ASR data files.
      - `ina.jsonl`: Converted ASR data for the INA dataset, in JSON Lines format.
  - `htr/`: Directory for Handwritten Text Recognition (HTR) data.
    - `original/`: Contains the original HTR data files.
      - `htrec/*csv`: Original HTR files for the HTRec dataset, in CSV format.
    - `converted/`: Contains converted HTR data files.
      - `htrec.jsonl`: Converted HTR data for the HTRec dataset, in JSON Lines format.
  - `ocr/`: Directory for Optical Character Recognition (OCR) data.
    - `original/`: Contains the original OCR data files.
      - `ajmc/*tsv`: Original OCR files for the AJMC dataset, in TSV format.
      - `icdar-2017/*txt`: Original OCR files for the ICDAR-2017 dataset, in text format.
      - `icdar-2019/*txt`: Original OCR files for the ICDAR-2019 dataset, in text format.
      - `impresso-nzz/*xml`: Original OCR files for the Impresso-NZZ dataset, in XML format.
      - `overproof/*txt`: Original OCR files for the Overproof dataset, in text format.
    - `converted/`: Contains converted OCR data files.
      - `ajmc.jsonl`: Converted OCR data for the AJMC dataset, in JSON Lines format.
      - `icdar-2017.jsonl`: Converted OCR data for the ICDAR-2017 dataset, in JSON Lines format.
      - `icdar-2019.jsonl`: Converted OCR data for the ICDAR-2019 dataset, in JSON Lines format.
      - `impresso-nzz.jsonl`: Converted OCR data for the Impresso-NZZ dataset, in JSON Lines format.
      - `overproof.jsonl`: Converted OCR data for the Overproof dataset, in JSON Lines format.
- `prompts/`: Directory containing various prompts used to guide or initiate specific processes.
  - `basic/`: Contains basic prompts for initial or simple tasks. Examples:
    - `prompt_basic_01.txt`
    - `prompt_basic_02.txt`
    Complex prompts for more advanced tasks.
    - `prompt_complex_01.txt`
    - `prompt_complex_02.txt`
    Prompts tailored to specific languages.
    - `prompt_complex_02_fr.txt`
    - `prompt_de.txt`
  - `few_shot/`: Contains prompts designed for few-shot for each dataset.
    - `htrec/`
    - `ina/`
  ...
---
## Data Format

### `filename`

- Represents the location of the dataset file within the project's directory structure.
- Example: `"../../data/datasets/ocr/original/overproof/dataset3/rawTextAndHumanCorrectionAndOverproofCorrectionTriples/allArticles.txt"`

### `dataset_name`

- The name of the dataset used for the project.
- Example: `"overproof"`

### `groundtruth`

- Contains the manually corrected version of the text, serving as the accurate reference.
- Includes subfields for different granularity levels such as `line`, `sentence`, and `region`, each holding the corresponding corrected text.

### `ocr`

- Contains the text generated by the OCR process before any corrections are made.
- Similar to `groundtruth`, it includes `line`, `sentence`, and `region` subfields with the OCR-generated text.

### `article_id`

- A unique identifier for the article, including metadata like publication year (`year`), article type (`type`), and a URL to the original source or PDF (`title`).

## Example of a JSON Line

A JSON line (`.jsonl` file) contains multiple JSON objects, each on a separate line. Here is a simplified example of what one line in a `.jsonl` file might look like for this dataset:

```json
{
  "filename": "data/ocr/original/overproof/sample.txt",
  "dataset_name": "overproof",
  "groundtruth": {
    "line": "historical event took",
    "sentence": "The historical event took place in the late 19th century.",
    "region": " The historical event took place in the late 19th century. 
                It was a significant occurrence during that period."
  },
  "ocr": {
    "line": "historLal evnt took",
    "sentence": "The historical evnt took place in the late 19th century.",
    "region": " The historical evnt took place in the late 19th century. 
                It ws a significant occurrence during that period."
  },
  "article_id": "year 1899 type Article title http://example.com/sample.pdf"
}
