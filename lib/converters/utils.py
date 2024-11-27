import pysbd
from genalog.text import anchor
import re
import warnings
warnings.filterwarnings('ignore')


def clean_text(text):
    """
    :param text:
    :return:
    """
    # Remove any "#" characters and extra spaces
    cleaned_text = re.sub(r"#+", "", text).strip()
    cleaned_text = re.sub(r"@+", "", cleaned_text).strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.replace('¬ ', '')
    cleaned_text = cleaned_text.replace('¬\n', '')

    # If ¬ is still in the sentence:
    cleaned_text = cleaned_text.replace('¬', '')
    return cleaned_text


def align_texts(gt_text, ocr_text, language='en'):
    gt_text = clean_text(gt_text)
    ocr_text = clean_text(ocr_text)

    try:
        segmenter = pysbd.Segmenter(language=language, clean=False)
    except BaseException:
        # Defaulting to en if a tokenizer is not available in a specific
        # language
        segmenter = pysbd.Segmenter(language='en', clean=False)

    # We align the texts with RETAS Method
    aligned_gt, aligned_noise = anchor.align_w_anchor(gt_text, ocr_text)

    # We split the ground truth sentences and we consider them as the
    # "correct" tokenization
    gt_sentences = segmenter.segment(aligned_gt)

    # We split the noisy text following the sentences' lengths in the ground
    # truth
    sentence_lengths = [len(sentence) for sentence in gt_sentences]

    ocr_sentences = []
    start = 0

    for length in sentence_lengths:
        end = start + length
        ocr_sentences.append(aligned_noise[start:end])
        start = end

    assert len(gt_sentences) == len(ocr_sentences)

    aligned_sentences = []
    # Clean the sentences from the alignment characters @
    for gt_sentence, ocr_sentence in zip(gt_sentences, ocr_sentences):
        aligned_sentences.append(
            (clean_text(gt_sentence), clean_text(ocr_sentence)))

    return aligned_sentences


# Function to reconstruct sentences from text lines
def reconstruct_text(txt_lines, sentences):
    reconstructed_text = " ".join(txt_lines)
    line_index_mapping = []
    sentence_index_mapping = []

    for line in txt_lines:
        start_index = reconstructed_text.find(line)

        if start_index != -1:
            end_index = start_index + len(line)
            line_index_mapping.append(
                {"line": line, "start_index": start_index, "end_index": end_index})

    for sentence in sentences:
        start_index = reconstructed_text.find(sentence)

        if start_index != -1:
            end_index = start_index + len(sentence)
            sentence_index_mapping.append(
                {"sentence": sentence, "start_index": start_index, "end_index": end_index})

    return reconstructed_text, line_index_mapping, sentence_index_mapping


def map_lines_to_sentences(lines, sentences, ocr_lines, ocr_sentences):
    line_index_mapping = {}
    sentence_index_mapping = {}
    result = []
    ocr_result = []
    for i, line in enumerate(lines):
        if i not in line_index_mapping:
            for j, sentence in enumerate(sentences):
                if line in sentence:
                    sentence_index_mapping[j] = sentence
                    result.append((line, sentence))
                    ocr_result.append((ocr_lines[i], ocr_sentences[j]))
                    break
                elif sentence in line:
                    sentence_index_mapping[j] = sentence
                    result.append((line, sentence))
                    ocr_result.append((ocr_lines[i], ocr_sentences[j]))
                    break

    for i, sentence in enumerate(sentences):
        sentence_index_mapping[i] = sentence
        start = 0
        for j, line in enumerate(lines):
            if sentence in line:
                line_index_mapping[j] = sentence
                result.append((line, sentence))
                ocr_result.append((ocr_lines[j], ocr_sentences[i]))
                start = len(line)
            elif start > 0 and line in sentence[start:]:
                line_index_mapping[j] = sentence
                result.append((line, sentence))
                ocr_result.append((ocr_lines[j], ocr_sentences[i]))
                start = 0
            elif sentence in line:
                line_index_mapping[j] = sentence
                result.append((line, sentence))
                ocr_result.append((ocr_lines[j], ocr_sentences[i]))

    return result, ocr_result
