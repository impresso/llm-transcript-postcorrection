# Experimental Setup

The methodology involves comparing the similarities between original automated transcriptions and their ground truth (GT), and between LLM-corrected versions and GT, to evaluate improvements or degradations, in the following setups:
* zero-shot
* few-shot (3-shot)
* basic/complex and language-specific prompts

### Models

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

### Repository Structure

```
lib/
├── converters/
│   ├── htrec_converter.py
│   ├── ina_converter.py
│   ├── icdar_converter.py
│   ├── nzz_converter.py
│   └── overproof_converter.py
├── modules/
│   └── dataset.py
├── main.py
└── prompt.py

```
- `lib/`: The root library directory.
  - `converters/`: Contains scripts for converting datasets from their original format to a standardized format.
    - `htrec_converter.py`: Converts the HTRec dataset.
    - `ina_converter.py`: Converts the INA dataset.
    - `icdar_converter.py`: Converts the ICDAR dataset.
    - `nzz_converter.py`: Converts the NZZ dataset.
    - `overproof_converter.py`: Converts the Overproof dataset.
  - `modules/`: Contains modules used across the project.
    - `dataset.py`: A module for handling dataset operations.
  - `main.py`: The main script that orchestrates the execution of conversions or other operations.
  - `prompt.py`: A script for generating prompts or handling input/output operations.
  
---
### Examples of Usage

The provided commands are examples of how to run our main experimental setup (`main.py`) with various parameters and flags for processing datasets using specific configurations. Each command sets up the environment and specifies options for dataset processing. Here's a detailed explanation of each command:

### Command 1: Basic Prompt Processing for HTREC Dataset

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python main.py \
        --input_dir ../data/datasets/ocr/converted/htrec\
        --output_dir ../data/output \
        --config_file ../data/config.yml \
        --prompt_dir ../data/prompts \
        --device cuda --prompt prompt_basic_02.txt
```

- `CUDA_VISIBLE_DEVICES=0`: Specifies the ID of the CUDA device to use, ensuring the script runs on the first GPU.
- `TOKENIZERS_PARALLELISM=false`: Disables parallel tokenization to avoid potential issues with multithreading in tokenizers.
- `--input_dir`: Path to the input directory containing the HTRec dataset that has been converted.
- `--output_dir`: Directory where the processed output will be saved.
- `--config_file`: Path to the YAML configuration file that specifies parameters for processing.
- `--prompt_dir`: Directory containing prompt files.
- `--device cuda`: Specifies that CUDA (GPU) should be used for processing.
- `--prompt`: The specific prompt file to use for processing, in this case, `prompt_basic_02.txt`.

### Command 2: Complex Prompt Processing for INA Dataset with Few-Shot Learning

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python main.py \
        --input_dir ../data/datasets/asr/converted/ina \
        --output_dir ../data/output/lang_specific/ \
        --config_file ../data/config_cluster.yml \
        --prompt_dir ../data/prompts \
        --device cuda --prompt prompt_complex_03_ina_fr.txt \
        --few-shot
```

- Similar to the first command, but processes the INA dataset from the ASR data.
- `--output_dir`: The output directory is now a subdirectory for language-specific outputs.
- `--config_file`: Uses a different configuration file (`config_cluster.yml`) that might contain cluster-specific settings or parameters for few-shot learning.
- `--prompt`: Specifies a complex, language-specific prompt file (`prompt_complex_03_ina_fr.txt`).
- `--few-shot`: Enables few-shot learning mode for processing, which utilizes a small number of examples to guide the model's processing or predictions.

### Command 3: Few-Shot, Language-Specific Processing for HTRec Dataset on CPU

```bash
TOKENIZERS_PARALLELISM=false python main.py \
        --input_dir ../data/datasets/htr/converted/htrec \
        --output_dir ../data/output/few_shot/prompt_complex_lang \
        --config_file ../data/config.yml \
        --prompt_dir ../data/prompts \
        --device cpu \
        --prompt prompt_complex_02.txt \
        --lang-specific \
        --few-shot
```

- Processes the HTREC dataset, similar to Command 1, but with different options.
- `--device cpu`: Specifies that the processing should be done on the CPU instead of a GPU.
- `--output_dir`: Output directory tailored for few-shot and language-specific processing.
- `--lang-specific`: Indicates that the processing should consider language-specific features or settings.
- `--few-shot`: Enables few-shot learning mode, as in Command 2.
