import unittest
from lib.converters.icdar_converter import (process_text,
                                            custom_similarity,
                                            align_sentences)

text = """[OCR_toInput] SECT III. WEEP NO MORE. 109 Whose passions not his masters are, Whose soul is still prepared
for death, Untied unto the worldly care Of public fame or private breath Who envies none that chance doth raise,
Or vice who never understood How deepest wounds are given by praise Nor rules of state, but rules of good Who hath
his life from rumours freed, Whose conscience is his strong retreat Whose state can neither flatterers feed,
Nor ruin make accusers great Who God doth late and early pray More of His grace than gifts to lend And entertains the
harmless day With a religious book or friend. This man is freed from servile bands Of hope to rise, or fear to fall
Lord of himself, though not of lands And having nothing, yet hath all. 1614. - Wotton. xcvi WEEP NO MORE. Weep no
more, nor sigh, nor groan, Sorrow calls no time that's gone Violets plucked the sweetest rain Makes not fresh nor
grow again Trim thy locks, look cheerfully Fate's hidden ends eyes cannot see Toys as winged dreams fly fast,
Why should sadness longer last ? Grief is but a wound to woe Gentlest fair, mourn, mourn no more. - Fletcher. 1614.
From The Queen of Corinth. [OCR_aligned] SECT III. WEEP NO MORE. 109 Whose passions not his masters are, Whose soul
is still prepared for death, Untied unto the worldly care Of public fame or private breath Who envies none that
chance doth raise, Or vice who never understood How deepest wounds are given by praise Nor rules of state,
but rules of good Who hath his life from rumours freed, Whose conscience is his strong retreat Whose state can
neither flatterers feed, Nor ruin make accusers great Who God doth late and early pray More of His grace than gifts
to lend And entertains the harmless day With a religious book or friend. This man is freed from servile bands Of hope
to rise, or fear to fall Lord of himself, though not of lands And having nothing, yet hath all. 1614. - Wotton. xcvi@
WEEP NO MORE. Weep no more, nor sigh, nor groan, Sorrow calls no time that's gone Violets plucked the sweetest rain
Makes not fresh nor grow again Trim thy locks, look cheerfully Fate's hidden ends eyes cannot see Toys as winged
dreams fly fast, Why should sadness longer last ? Grief is but a wound to woe Gentlest fair, mourn, mourn no more. -
Fletcher. 1614. From The Queen of Corinth. [ GS_aligned] ################################# passions not his masters
are, Whose soul is still prepared for death, Untied unto the worldly care Of public fame or private breath Who envies
none that chance doth raise, Or vice who never understood How deepest wounds are given by praise Nor rules of state,
but rules of good Who hath his life from rumours freed, Whose conscience is his strong retreat Whose state can
neither flatterers feed, Nor ruin make accusers great Who God doth late and early pray More of His grace than gifts
to lend And entertains the harmless day With a religious book or friend. This man is freed from servile bands Of hope
to rise, or fear to fall Lord of himself, though not of lands And having nothing, yet hath all. 1614. -@Wotton. XCVI.
WEEP NO MORE. WEEP no more, nor sigh, nor groan, Sorrow calls no time that's gone Violets plucked the sweetest rain
Makes not fresh nor grow again Trim thy locks, look cheerfully Fate's hidden ends eyes cannot see Toys as winged
dreams fly fast, Why should sadness longer last ? Grief is but a wound to woe Gentlest fair, mourn, mourn no more.
-@Fletcher. 1614. From The Queen of Corinth."""


class TestAlignSentences(unittest.TestCase):

    def test_align_sentences(self):

        texts = text.split('\n')
        ocr_sentences = process_text(
            texts[0].replace(
                '[OCR_toInput]',
                '').strip())
        gs_sentences = process_text(
            texts[1].replace(
                '[ GS_aligned]',
                '').strip())

        aligned_sentences = align_sentences(ocr_sentences, gs_sentences)

        # Check if the lengths of the input and aligned sentences are equal
        self.assertEqual(len(ocr_sentences), len(aligned_sentences))

        # Check if the custom distance is smaller for aligned sentences than
        # for non-aligned sentences
        for i, (ocr_sentence, gs_sentence) in enumerate(aligned_sentences):
            aligned_distance = custom_similarity(ocr_sentence, gs_sentence)
            for j, other_gs_sentence in enumerate(gs_sentences):
                if j != i:
                    other_distance = custom_similarity(
                        ocr_sentence, other_gs_sentence)
                    self.assertLessEqual(aligned_distance, other_distance)


if __name__ == "__main__":
    unittest.main()
