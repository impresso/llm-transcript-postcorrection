from utils import clean_text, align_texts
from const import Const
import os
import argparse
import logging
from tqdm import tqdm
import json
from langdetect import detect
import re
import sys
main_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(main_dir)


def remove_tags(text):
    text = re.sub(r'<[^>]+>', '', text)
    return text


def repair_punctuation(text):
    cleaned_text = re.sub(r'\s+([,.:;?!"\'])', r'\1', text)
    return cleaned_text


def process_file(args,
                 input_file: str,
                 output_file: str,
                 dataset_name: str) -> None:

    # Parse the ground truth file
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    # Remove the NE tags
    text = remove_tags(text)
    language = detect(text)

    lines = [clean_text(line) for line in text.split('\n')]

    # The lines are annotated with entities, where entities are Uppercased
    # thus, the ASR sentence is lowercase
    aligned_lines = []
    for line in lines:
        aligned_lines.append((line, line.lower()))

    # The region in Quaero is the article level
    gt_region_text = ' '.join(
        [gt_line for gt_line, _ in aligned_lines]).strip()
    ocr_region_text = ' '.join(
        [ocr_line for _, ocr_line in aligned_lines]).strip()

    aligned_sentences = align_texts(gt_region_text, ocr_region_text,
                                    language=language)

    gt_lines, gt_sentences, ocr_lines, ocr_sentences = [
        gt_line for gt_line, _ in aligned_lines], [
        gt_sentence for gt_sentence, _ in aligned_sentences], [
            ocr_line for _, ocr_line in aligned_lines], [
                ocr_sentence for _, ocr_sentence in aligned_sentences]

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
    # Create the mapping list

    # Append the output to a JSON Lines file
    with open(output_file, "a") as outfile:
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
                                    Const.GROUND: {Const.LINE: repair_punctuation(clean_text(gt_line)),
                                                   Const.SENTENCE: repair_punctuation(clean_text(gt_sentence)),
                                                   Const.REGION: repair_punctuation(clean_text(gt_region_text))}
                                    # TODO removed temporarily the region - too
                                    # large
                                    })
            outfile.write(json_line + "\n")
            outfile.flush()


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

    total_files = sum([len([file for file in files if file.endswith(".ne")])
                       for r, d, files in os.walk(args.input_dir)])
    progress_bar = tqdm(
        total=total_files,
        desc="Processing files",
        unit="file")

    output_dir_path = args.input_dir.replace('original', 'converted')

    dataset_name = args.input_dir.split('/')[-1]
    output_file = os.path.join(
        args.output_dir,
        '{}.jsonl'.format(dataset_name).lower())
    if os.path.exists(output_file):
        logging.info('{} already exists. It will be deleted.')
        os.remove(output_file)

    logging.info('Writing output {}'.format(output_file))
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".ne"):
                input_file = os.path.join(root, file)

                logging.info('Analyzing file {}'.format(input_file))

                process_file(
                    args=args,
                    input_file=input_file,
                    output_file=output_file,
                    dataset_name=dataset_name)
                progress_bar.update(1)
    progress_bar.close()
