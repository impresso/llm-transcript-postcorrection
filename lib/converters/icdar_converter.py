from sklearn.model_selection import train_test_split
import warnings
import pandas as pd
import glob
from utils import clean_text, align_texts
from const import Const
import os
import json
import argparse
from tqdm import tqdm
import logging
from langdetect import detect
import sys
main_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(main_dir)
warnings.filterwarnings('ignore')


def load_metadada(args):
    metadata_path = None
    for path in glob.glob(args.input_dir + '/**/*', recursive=True):
        if 'eval_metadata' in path:
            metadata_path = path

    if metadata_path is None:
        print('Metadata was not found.')
        args.metadata = None
    else:
        with open(metadata_path, 'r') as f:
            metadata = f.readlines()

        metadata = [line.strip().split(';') for line in metadata]
        # Skip columns line: 'File;Date;Type;NbAlignedChar'
        columns, metadata = metadata[0], metadata[1:]

        metadata = pd.DataFrame(metadata, columns=columns)
        metadata['File'] = metadata['File'].apply(
            lambda x: x.replace(
                '\\', '/'))  # Replace Windows style file names

        args.metadata = metadata


def lookup_metadata(args, input_file):
    file_metadata = args.metadata[args.metadata.File ==
                                  '/'.join(input_file.split('/')[-2:])]
    return file_metadata.to_dict('records')[0]


def process_file(args, input_file, output_file, dataset_name):
    # Read the input file
    with open(input_file, "r") as infile:
        data = infile.readlines()

    if args.metadata is not None:
        file_metadata = lookup_metadata(args, input_file)
    else:
        file_metadata = {}

    # Extract OCR and GS sentences from the data list
    # [OCR_toInput] [OCR_aligned] [ GS_aligned]
    gt_text = clean_text(data[2].replace('[ GS_aligned]', '').strip())
    ocr_text = clean_text(data[0].replace('[OCR_toInput]', '').strip())

    language = input_file.split('/')[-3].lower()

    try:
        # Align the OCR and GS sentences
        aligned_sentences = align_texts(gt_text, ocr_text, language=language)
    except BaseException:
        # Defaulting to English
        aligned_sentences = align_texts(gt_text, ocr_text, language='en')

    # Write the output to a JSON Lines file
    with open(output_file, "a") as outfile:
        for ocr_sentence, gs_sentence in aligned_sentences:
            json_line = json.dumps({Const.LANGUAGE: input_file.split('/')[-3].lower(),
                                    Const.FILE: input_file,
                                    Const.DATASET: dataset_name,
                                    Const.OCR: {Const.LINE: Const.NONE,
                                                Const.SENTENCE: ocr_sentence,
                                                Const.REGION: ocr_text},  # TODO removed temporarily the region - too large
                                    Const.GROUND: {Const.LINE: Const.NONE,
                                                   Const.SENTENCE: gs_sentence,
                                                   Const.REGION: gt_text}  # TODO removed temporarily the region - too large
                                    } | file_metadata)

            outfile.write(json_line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process text files and align sentences.")
    parser.add_argument(
        "--input_dir",
        help="The path to the input directory containing the text files.")
    parser.add_argument(
        "--output_dir",
        help="The path to the output directory where JSON Lines files will be created.")
    parser.add_argument(
        "--language", default='de',
        help="The language of the dataset.")
    parser.add_argument(
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    args = parser.parse_args()

    load_metadada(args)

    output_dir_path = args.input_dir.replace('original', 'converted')

    dataset_name = args.input_dir.split('/')[-1]
    output_file = os.path.join(args.output_dir,
                               '{}.jsonl'.format(dataset_name))
    if os.path.exists(output_file):
        logging.info('{} already exists. It will be deleted.')
        os.remove(output_file)

    import glob

    files, langs = [], []
    logging.info('Writing output {}'.format(output_file))
    for input_file in glob.glob(f"{args.input_dir}/**/*", recursive=True):
        if not os.path.isdir(input_file):
            # do something with the file
            if input_file.endswith(".txt") and 'readme' not in input_file:

                logging.info('Analyzing file {}'.format(input_file))
                if os.path.getsize(input_file) / 1024 <= 40:
                    # print(input_file, os.path.getsize(input_file) / 1024)
                    files.append(input_file)
                    langs.append(input_file.split('/')[-3])

    if len(files) > 200:
        files_keep, files_removed, _, _ = train_test_split(
            files, langs, test_size=0.986, random_state=43)
        print(len(files_keep), len(files_removed))
        files = files_keep
    else:
        # icdar 2017
        files_keep, files_removed, _, _ = train_test_split(
            files, langs, test_size=0.85, random_state=43)
        print(len(files_keep), len(files_removed))
        files = files_keep
    # total_files = sum([len(files) for r, d, files in os.walk(args.input_dir)])
    total_files = len(files)
    progress_bar = tqdm(
        total=total_files,
        desc="Processing files",
        unit="file")

    print(f'There are {len(files)} files')
    for input_file in files:
        process_file(
            args=args,
            input_file=input_file,
            output_file=output_file,
            dataset_name=dataset_name)
        progress_bar.update(1)
    progress_bar.close()

    # TRAIN SET
    output_file = os.path.join(args.output_dir,
                               '{}-train.jsonl'.format(dataset_name))
    if os.path.exists(output_file):
        logging.info('{} already exists. It will be deleted.')
        os.remove(output_file)

    total_files = len(files_removed)
    progress_bar = tqdm(
        total=total_files,
        desc="Processing files",
        unit="file")

    print(f'There are {len(files_removed)} files')
    for input_file in files_removed:
        process_file(
            args=args,
            input_file=input_file,
            output_file=output_file,
            dataset_name=dataset_name)
        progress_bar.update(1)
    progress_bar.close()
