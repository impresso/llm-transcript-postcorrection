from utils import clean_text, align_texts
from const import Const
import os
from bs4 import BeautifulSoup
import argparse
import logging
from tqdm import tqdm
import json
import sys
main_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(main_dir)


def process_file(args,
                 input_file: str,
                 ocr_file: str,
                 output_file: str,
                 dataset_name: str) -> None:

    # Parse the ground truth file
    with open(input_file, 'r') as f:
        gt_soup = BeautifulSoup(f, 'xml')

    aligned_texts = []
    # Parse the prediction file
    if not os.path.exists(ocr_file):
        print(f'{ocr_file} does not exist.')
    else:
        with open(ocr_file, 'r') as f:
            ocr_soup = BeautifulSoup(f, 'xml')

        # Iterate over each TextRegion in the ground truth file
        for gt_region in gt_soup.find_all('TextRegion'):
            # Find the corresponding TextRegion in the prediction file
            region_id = gt_region['id']
            ocr_region = ocr_soup.find('TextRegion', {'id': region_id})

            # Extract the TextEquiv content for the TextRegion in the ground
            # truth and prediction files
            gt_region_text = clean_text(
                gt_region.findAll('TextEquiv')[-1].text.strip())

            ocr_region_text = None
            try:
                ocr_region_text = clean_text(ocr_region.findAll(
                    'TextEquiv')[-1].text.strip())
            except BaseException as ex:
                print(f'{region_id} not found in {ocr_file}. Exception: {ex}')
            # Print the extracted TextEquiv content for the TextRegion in the
            # ground truth and prediction files

            # If the region was found
            if ocr_region_text:
                aligned_lines = []
                for gt_line in gt_region.find_all('TextLine'):
                    # Find the corresponding TextLine in the prediction file
                    line_id = gt_line['id']
                    ocr_line = ocr_region.find('TextLine', {'id': line_id})

                    # Extract the TextEquiv content for the TextLine in the
                    # ground truth and prediction files
                    gt_line_text = clean_text(
                        gt_line.findAll('TextEquiv')[-1].text.strip())

                    ocr_line_text = None
                    try:
                        ocr_line_text = clean_text(
                            ocr_line.findAll('TextEquiv')[-1].text.strip())
                        # ocr_line_text = ocr_line.find('TextEquiv').text.strip()
                    except BaseException as ex:
                        print(
                            f'Line {line_id} not found in {ocr_file} in region {region_id}. Exception: {ex}')

                    # Print the extracted TextEquiv content for the TextLine in the ground truth and
                    # prediction files
                    if ocr_line_text:
                        # import pdb;pdb.set_trace()
                        aligned_lines.append((gt_line_text, ocr_line_text))

                if ocr_region_text:
                    # Add the already aligned regions
                    aligned_texts += [(gt_region_text, ocr_region_text)]

                if ocr_region_text:
                    # Align the OCR and GS sentences
                    aligned_sentences = align_texts(
                        gt_region_text, ocr_region_text, language=args.language)

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
                                                # TODO removed temporarily the
                                                # region - too large
                                                Const.GROUND: {Const.LINE: clean_text(gt_line),
                                                               Const.SENTENCE: clean_text(gt_sentence),
                                                               Const.REGION: clean_text(gt_region_text)}
                                                # TODO removed temporarily the
                                                # region - too large
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
        '--ocr_dir',
        type=str,
        help='Path to OCRed folder',
        default='../../data/datasets/ocr/original/impresso-nzz/xml/ABBYY_FineReader_Server11')
    parser.add_argument(
        '--test_file_name',
        type=str,
        help='Path to OCRed folder',
        default='../../data/datasets/ocr/original/impresso-nzz/test-set-filenames.txt')
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

    total_files = sum([len([file for file in files if file.endswith(".xml")])
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
    if len(dataset_name.strip()) == 0:
        dataset_name = args.input_dir.split('/')[-2]
        output_file = os.path.join(
            args.output_dir,
            '{}.jsonl'.format(dataset_name).lower())
    if os.path.exists(output_file):
        logging.info('{} already exists. It will be deleted.')
        os.remove(output_file)

    with open(args.test_file_name, 'r') as f:
        test_set_filenames = [x.strip() for x in f.readlines()]

    import glob
    gt_files, ocr_files = [], []
    logging.info('Writing output {}'.format(output_file))
    for root, dirs, files in os.walk(args.input_dir):
        for input_file in files:
            # do something with the file
            file = os.path.join(root, input_file)
            # print(file)
            ocr_file = os.path.join(args.ocr_dir, input_file)

            if file.endswith(".xml") and 'readme' not in file:

                # logging.info('Analyzing file {}'.format(file))
                if input_file in test_set_filenames:
                    gt_files.append(file)
                    ocr_files.append(ocr_file)
                else:
                    if os.path.exists(os.path.join(root, file)):
                        print(f'Deleting {file}.')
                        os.remove(os.path.join(root, file))
                        if os.path.exists(os.path.join(args.ocr_dir, file)):
                            os.remove(os.path.join(args.ocr_dir, file))

    print(len(gt_files), len(ocr_files))

    from sklearn.model_selection import train_test_split
    files_keep, files_removed, ocr_files_keep, ocr_files_removed = train_test_split(
        gt_files, ocr_files, test_size=0.5, random_state=43)
    print(len(files_keep), len(files_removed))

    logging.info('Writing output {}'.format(output_file))

    for input_file, ocr_file in zip(files_keep, ocr_files_keep):
        # if file in test_set_filenames:
        # input_file = os.path.join(root, file)
        # ocr_file = os.path.join(args.ocr_dir, file)
        # print(input_file)
        logging.info('Analyzing file {}'.format(input_file))

        process_file(
            args=args,
            input_file=input_file,
            ocr_file=ocr_file,
            output_file=output_file,
            dataset_name=dataset_name)
        progress_bar.update(1)

    # Removing the OCRed files that are not in test-set-filenames.
    # There are more than the groundtruth.
    for root, dirs, files in os.walk(args.ocr_dir):
        for file in files:
            if file.endswith(".xml"):
                if file not in test_set_filenames:
                    print(f'Deleting {file}.')
                    if os.path.exists(os.path.join(args.ocr_dir, file)):
                        os.remove(os.path.join(args.ocr_dir, file))
    progress_bar.close()
