# -*- coding: utf-8 -*-
from torch import nn
import openai
import yaml
import os
import argparse
from tqdm import tqdm
import importlib
import jsonlines
import logging
from const import Const
from retrying import retry
logger = logging.getLogger("gpt-experiments")
logger.setLevel(logging.INFO)

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_dict(list_of_dicts: list):
    return_list = {}
    for list_entry in list_of_dicts:
        return_list |= list_entry
    return return_list


def generate(
        input_dir: str = "../data/datasets",
        output_dir: str = "../data/output",
        prompt_dir: str = "../data/prompts",
        config_file: str = "../data/config.yml",
        few_shot: bool = True,
        lang_specific: bool = True,
        device: str = 'cpu'
) -> None:
    """
    Generates texts via several models in config and saves them to predictions files.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    openai.api_key = os.getenv("OPENAI_API_KEY")  # config['SECRET_KEY']

    prompt_path = os.path.join(prompt_dir, args.prompt)
    # If prompt is a file path, load the file as the prompt.
    print(prompt_dir, args.prompt)
    if os.path.exists(prompt_path):

        logger.info(f"Loading prompt from {prompt_path}.")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()
            print(prompt)
    else:
        logger.info(
            f"Model prompt missing: {prompt_path}. The prompt will be loaded dynamically.")

    for model in config['models']:

        (model_name, experiment_details), = model.items()

        logger.info('Experimenting with {}'.format(model_name))

        experiment_details = get_dict(experiment_details)

        model_class = experiment_details['class']
        # prompt_path = os.path.join(prompt_dir, experiment_details['prompt'])
        print(few_shot)
        if few_shot and lang_specific:
            results_dir = os.path.join(
                output_dir, 'few_shot/prompt_complex_lang/',
                args.prompt.replace(
                    '.txt',
                    ''))
        elif few_shot:
            # import pdb;
            # pdb.set_trace()
            results_dir = os.path.join(
                output_dir, 'few_shot',
                args.prompt.replace(
                    '.txt',
                    ''))
            print(results_dir)
        else:
            results_dir = os.path.join(
                output_dir,
                args.prompt.replace('.txt', ''))
            print(results_dir)
            # import pdb;pdb.set_trace()
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        module = importlib.import_module('prompt')
        class_ = getattr(module, model_class)
        instance = class_(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            device=device)

        # Iterate in the data folder with all datasets
        print('WHA', input_dir)
        for root, dirs, files in os.walk(input_dir, topdown=False):
            for name in files:
                if name.endswith('jsonl'):
                    input_file = os.path.join(root, name)

                    logging.info(f'Post-correcting {input_file}')
                    dataset_name = name.replace('.jsonl', '')
                    print('-----', input_file, dataset_name)

                    dataset_model_results_dir = os.path.join(
                        results_dir, dataset_name)
                    if not os.path.exists(dataset_model_results_dir):
                        os.makedirs(dataset_model_results_dir)

                    if few_shot:
                        print('BY HERE')
                        output_file = os.path.join(
                            dataset_model_results_dir,
                            'results-3few-shot-{}-{}.jsonl'.format(
                                dataset_name,
                                model_name).replace(
                                '/',
                                '-'))
                    else:
                        output_file = os.path.join(
                            dataset_model_results_dir,
                            'results-{}-{}.jsonl'.format(
                                dataset_name,
                                model_name).replace(
                                '/',
                                '-'))

                    logger.info(
                        'Predictions for {} are saved in {}'.format(
                            input_file, output_file))

                    with open(input_file, 'r') as g:
                        total_files = len(g.readlines())

                    progress_bar = tqdm(
                        total=total_files,
                        desc="Processing files",
                        unit="file")

                    mode = 'w'
                    count = None

                    if os.path.exists(output_file):
                        mode = 'a'
                        logging.info(
                            'We found the results file {}. We will continue predicting from where it was '
                            'left off.'.format(output_file))
                        lines = []
                        count = 0
                        with jsonlines.open(output_file, 'r') as f:
                            for json_line in f:
                                lines.append(json_line)
                                count += 1

                    @retry(stop_max_attempt_number=20,
                           wait_exponential_multiplier=1000,
                           wait_exponential_max=10000)
                    def get_prediction(prompt, model_name):
                        options = {
                            'engine': model_name,
                            'top_p': 1.0,
                            'frequency_penalty': 0,
                            'presence_penalty': 0
                        }
                        return instance.prediction(prompt, options)

                    already_done = {}

                    with jsonlines.open(output_file, mode) as f:
                        with jsonlines.open(input_file, 'r') as g:
                            for idx, json_line in enumerate(g):
                                progress_bar.update(1)
                                if count is not None:
                                    if idx <= count:
                                        continue

                                data = {
                                    Const.PREDICTION: {
                                    }
                                }

                                for TEXT_LEVEL in [
                                        Const.LINE, Const.SENTENCE, Const.REGION]:
                                    text = json_line[Const.OCR][TEXT_LEVEL]
                                    if text not in already_done:
                                        if text is not None:
                                            # import pdb;pdb.set_trace()
                                            if 'ajmc' in dataset_name:
                                                language = 'el'
                                            elif 'overproof' in dataset_name:
                                                language = 'en'
                                            elif 'impresso' in dataset_name:
                                                language = 'de'
                                            elif 'htrec' in dataset_name:
                                                language = 'el'
                                            elif 'ina' in dataset_name:
                                                language = 'fr'
                                            elif 'icdar-2017' in dataset_name:
                                                language = json_line['filename'].split(
                                                    '/')[-2].split('_')[0]
                                                if language == 'eng':
                                                    language = 'en'
                                            else:
                                                language = json_line['language']

                                            if (few_shot) and (lang_specific):

                                                prompt_path = os.path.join(prompt_dir, 'few_shot_lang', dataset_name.replace(
                                                    '_', '-'), f'{args.prompt.replace(".txt", "")}_{TEXT_LEVEL}_{language}.txt')

                                                if os.path.exists(prompt_path):
                                                    # logger.info(f"---Loading prompt from {prompt_path}.")
                                                    with open(prompt_path, "r", encoding="utf-8") as g:
                                                        prompt = g.read()
                                                else:
                                                    logger.info(
                                                        f"----Model prompt missing: {prompt_path}.")

                                                data[Const.PREDICTION][Const.PROMPT] = prompt.replace(
                                                    '{{TEXT}}', text)

                                            # attention for the few-shot
                                            # scenario
                                            elif few_shot:
                                                # TODO: lack of time,
                                                # workaround here in few-shot
                                                # ==> transform it temporarily
                                                # to lang-specific
                                                prompt_path = os.path.join(prompt_dir, 'few_shot', dataset_name.replace(
                                                    '_', '-'), f'{args.prompt.replace(".txt", "")}_{TEXT_LEVEL}_{language}.txt')

                                                if os.path.exists(prompt_path):
                                                    # logger.info(f"---Loading prompt from {prompt_path}.")
                                                    with open(prompt_path, "r", encoding="utf-8") as g:
                                                        few_shot_prompt = g.read()
                                                else:
                                                    logger.info(
                                                        f"----Model prompt missing: {prompt_path}.")

                                                data[Const.PREDICTION][Const.PROMPT] = few_shot_prompt.replace(
                                                    '{{TEXT}}', text)

                                            elif (lang_specific) and os.path.exists(prompt_path):
                                                logger.info(
                                                    f"---Loading prompt from {prompt_path}.")
                                                with open(prompt_path, "r", encoding="utf-8") as g:
                                                    lang_prompt = g.read()
                                                data[Const.PREDICTION][Const.PROMPT] = lang_prompt.replace(
                                                    '{{TEXT}}', text)

                                            elif lang_specific and (not os.path.exists(prompt_path)):
                                                prompt_path = os.path.join(
                                                    prompt_dir, f'prompt_complex_02_{language}.txt')
                                                if os.path.exists(prompt_path):
                                                    # logger.info(f"---Loading prompt from {prompt_path}.")
                                                    with open(prompt_path, "r", encoding="utf-8") as g:
                                                        lang_prompt = g.read()
                                                else:
                                                    logger.info(
                                                        f"----Model prompt missing: {prompt_path}.")

                                                # else it takes the prompt from
                                                # the arguments
                                                data[Const.PREDICTION][Const.PROMPT] = lang_prompt.replace(
                                                    '{{TEXT}}', text)
                                            else:
                                                data[Const.PREDICTION][Const.PROMPT] = prompt.replace(
                                                    '{{TEXT}}', text)

                                            result = get_prediction(
                                                data[Const.PREDICTION][Const.PROMPT], model_name)

                                            data[Const.PREDICTION][TEXT_LEVEL] = result
                                            already_done[text] = result

                                            # print(result)
                                    else:
                                        data[Const.PREDICTION].update(
                                            {TEXT_LEVEL: already_done[text]})

                                data = json_line | data
                                f.write(data)
                                # f.flush()

                                # data[Const.PREDICTION].update({Const.PROMPT: None})

                    progress_bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Base folder with input files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Base folder for prediction files.",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        help="Base folder with prompts data.",
    )
    parser.add_argument(
        "--device",
        default='cpu',
        type=str,
        help="The inference is done either on cuda or cpu.",
    )
    parser.add_argument(
        "--prompt",
        default='prompt_basic_02.txt',
        type=str,
        help="The selected prompt.",
    )
    parser.add_argument(
        "--config_file",
        default='../data/config.yml',
        type=str,
        help="The selected prompt.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    parser.add_argument(
        "--few-shot",
        help="few-shot",
        action="store_true",
    )
    parser.add_argument(
        "--lang-specific",
        help="lang-specific",
        action="store_true",
    )

    args = parser.parse_args()
    print(args)

    generate(args.input_dir,
             args.output_dir,
             args.prompt_dir,
             args.config_file,
             args.few_shot,
             args.lang_specific,
             args.device)
