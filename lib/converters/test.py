import textdistance
import nltk
from nltk.tokenize import sent_tokenize


def process_text(text):
    cleaned_text = text.strip()
    sentences = sent_tokenize(cleaned_text)
    return sentences


def custom_similarity(sentence1, sentence2):
    return textdistance.jaccard(sentence1, sentence2)


def align_texts(text1, text2):
    sentences1 = process_text(text1)
    sentences2 = process_text(text2)

    aligned_sentences = []
    for sentence1 in sentences1:
        best_match = (None, 0)
        for sentence2 in sentences2:
            similarity = custom_similarity(sentence1, sentence2)
            if similarity > best_match[1]:
                best_match = (sentence2, similarity)
        aligned_sentences.append((sentence1, best_match[0]))

    return aligned_sentences


text1 = "This is the first text. It has multiple sentences. Some are short. Some are long."
text2 = "This is the second text. It also has multiple sentences. Some are short, others are long."

aligned_sentences = align_texts(text1, text2)

for sentence1, sentence2 in aligned_sentences:
    print(f"Text1: {sentence1}\nText2: {sentence2}\n")
