
import pandas as pd
from utils import clean_text, align_texts
from sklearn.model_selection import train_test_split
from const import Const
import os
import argparse
import logging
from tqdm import tqdm
import json
from langdetect import detect
# Add the main directory to the sys.path list
import sys
main_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(main_dir)


def process_file(
        input_file: str,
        output_file: str,
        dataset_name: str) -> None:

    # Parse the ground truth file
    df = pd.read_csv(input_file)

    grouped = df.groupby('ImageID')

    # for name, group in grouped:
    #     print(name, group)
    # import pdb;pdb.set_trace()
    # HUMAN_TRANSCRIPTION, SYSTEM_TRANSCRIPTION, ImageID, TEXT_LINE_NUM, CENTURY

    aligned_texts = []

    language = 'el'

    # articles_keep, articles_removed, _, _ = train_test_split(articles, articles, test_size=0.8, random_state=43)

    for name, article in grouped:
        # Keep the article id
        article_id = name.strip()
        century = article.CENTURY.unique()[0]
        # Align the lines before all types of extraction so the region/article
        # can be produced
        aligned_lines = []
        for gt_line, ocr_line in zip(
                article.HUMAN_TRANSCRIPTION, article.SYSTEM_TRANSCRIPTION):
            aligned_lines.append((gt_line, ocr_line))

        gt_region_text = ' '.join(
            [gt_line[0] for gt_line in aligned_lines]).strip()
        ocr_region_text = ' '.join(
            [ocr_line[-1] for ocr_line in aligned_lines]).strip()
        aligned_texts.append((gt_region_text, ocr_region_text, article_id))

        # Split in sentences and align
        aligned_sentences = align_texts(gt_region_text,
                                        ocr_region_text,
                                        language=language)

        gt_lines, gt_sentences, ocr_lines, ocr_sentences = [gt_line[0] for gt_line in aligned_lines], \
            [gt_sentence for gt_sentence, _ in aligned_sentences], \
            [ocr_line[-1] for ocr_line in aligned_lines], \
            [ocr_sentence for _, ocr_sentence in aligned_sentences]

        # print(gt_lines, gt_sentences)

        from utils import map_lines_to_sentences
        gt_reconstructed_sentences, ocr_reconstructed_sentences = map_lines_to_sentences(
            gt_lines, gt_sentences, ocr_lines, ocr_sentences)

        try:
            assert len(gt_reconstructed_sentences) == len(
                ocr_reconstructed_sentences)
        except BaseException:
            import pdb
            pdb.set_trace()

        # Append the output to a JSON Lines file
        with open(output_file, "a", encoding='utf8') as outfile:
            for gt_element, ocr_element in zip(
                    gt_reconstructed_sentences, ocr_reconstructed_sentences):
                (gt_line, gt_sentence) = gt_element
                (ocr_line, ocr_sentence) = ocr_element

                json_line = json.dumps({Const.FILE: input_file,
                                        Const.DATASET: dataset_name,
                                        Const.OCR: {Const.LINE: clean_text(ocr_line),
                                                    Const.SENTENCE: clean_text(ocr_sentence),
                                                    Const.REGION: clean_text(ocr_region_text)},
                                        # TODO removed temporarily the region - too
                                        # large
                                        Const.GROUND: {Const.LINE: clean_text(gt_line),
                                                       Const.SENTENCE: clean_text(gt_sentence),
                                                       Const.REGION: clean_text(gt_region_text)}
                                        # TODO removed temporarily the region - too
                                        # large
                                        } | {'article_id': article_id, 'century': str(century)})
                outfile.write(json_line + "\n")
                outfile.flush()
        # import jsonlines
        # with jsonlines.open(output_file, 'r') as g:
        #     for x in g:
        #         # import pdb;pdb.set_trace()
        #         print(x)


if __name__ == "__main__":
    # Create an argument parser for the ground truth and OCR paths
    parser = argparse.ArgumentParser(
        description='Extract TextEquiv content for ground truth and prediction files.')
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Path to ground truth folder')
    parser.add_argument(
        "--output_dir",
        help="The path to the output directory where JSON Lines files will be created.")
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

    total_files = sum([len([file for file in files if file.endswith(".txt")])
                      for r, d, files in os.walk(args.input_dir)])
    progress_bar = tqdm(
        total=total_files,
        desc="Processing files",
        unit="file")

    output_dir_path = args.input_dir.replace('original', 'converted')

    dataset_name = args.input_dir.split('/')[-1]
    output_file = os.path.join(args.output_dir,
                               '{}.jsonl'.format(dataset_name))
    if os.path.exists(output_file):
        logging.info('{} already exists. It will be deleted.')
        os.remove(output_file)

    logging.info('Writing output {}'.format(output_file))
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            print(file)
            if file == "original_test.csv":
                input_file = os.path.join(root, file)

                logging.info('Analyzing file {}'.format(input_file))

                process_file(
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

    logging.info('Writing output {}'.format(output_file))
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            print(file)
            if file == "train.csv":
                input_file = os.path.join(root, file)

                logging.info('Analyzing file {}'.format(input_file))

                process_file(
                    input_file=input_file,
                    output_file=output_file,
                    dataset_name=dataset_name)
                progress_bar.update(1)
    progress_bar.close()
