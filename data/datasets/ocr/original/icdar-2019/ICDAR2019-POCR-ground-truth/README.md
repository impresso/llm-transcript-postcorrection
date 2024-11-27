Corpus for the ICDAR2019 Competition on Post-OCR Text Correction (October 2019)
=> Website: http://l3i.univ-larochelle.fr/ICDAR2019PostOCR
-------------------------------------------------------------------------------

#### Description:
The corpus accounts for 22M OCRed characters along with the corresponding Gold Standard (GS). The documents come from different digital collections available, among others, at the National Library of France (BnF) and the British Library (BL). The corresponding GS comes both from BnF's internal projects and external initiatives such as Europeana Newspapers, IMPACT, Project Gutenberg, Perseus and Wikisource.

Repartition of the dataset:
- **ICDAR2019_Post_OCR_correction_training_18M.zip** : 80% of the full dataset, provided to train participants' methods. 
- **ICDAR2019_Post_OCR_correction_evaluation_4M** : 20% of the full dataset used for the evaluation (with Gold Standard made publicly after the competition).
- **ICDAR2019_Post_OCR_correction_full_22M** : full dataset made publicly available after the competition.

#### Special case for Finnish language
Material from the National Library of Finland (Finnish dataset FI > FI1) are not allowed to be re-shared on other website. Please follow these guidelines to get and format the data from the original website.

1. Go to [https://digi.kansalliskirjasto.fi/opendata/submit?set_language=en](https://digi.kansalliskirjasto.fi/opendata/submit?set_language=en);
2. Download *OCR Ground Truth Pages (Finnish Fraktur) [v1](4.8GB)* from *Digitalia (2015-17)* package;
3. Convert the Excel file `~/metadata/nlf_ocr_gt_tescomb5_2017.xlsx` as Comma Separated Format (.csv) by using *save as* function in a spreadsheet software (e.g. Excel, Calc) and copy it into `FI/FI1/HOWTO_get_data/input/`;
4. Go to `FI/FI1/HOWTO_get_data/` and run `script_1.py` to generate the *full* `FI1` dataset in `output/full/`;
4. Run `script_2.py` to split the `output/full/` dataset into `output/training/` and `output/evaluation/` sub sets.
At the end of the process, you should have a `training`, `evaluation` and `full` folder with 1579528, 380817 and 1960345 characters respectively.


#### Licenses: free to use for non-commercial uses, according to sources in details:
- BG1: IMPACT - National Library of Bulgaria: CC BY NC ND
- CZ1: IMPACT - National Library of the Czech Republic: CC BY NC SA
- DE1: Front pages of Swiss newspaper NZZ: Creative Commons Attribution 4.0 International (https://zenodo.org/record/3333627)
- DE2: IMPACT - German National Library: CC BY NC ND
- DE3: GT4Hist-dta19 dataset: CC-BY-SA 4.0 (https://zenodo.org/record/1344132)
- DE4: GT4Hist - EarlyModernLatin: CC-BY-SA 4.0 (https://zenodo.org/record/1344132)
- DE5: GT4Hist - Kallimachos: CC-BY-SA 4.0 (https://zenodo.org/record/1344132)
- DE6: GT4Hist - RefCorpus-ENHG-Incunabula: CC-BY-SA 4.0 (https://zenodo.org/record/1344132)
- DE7: GT4Hist - RIDGES-Fraktur: CC-BY-SA 4.0 (https://zenodo.org/record/1344132)
- EN1: IMPACT - British Library: CC BY NC SA 3.0
- ES1: IMPACT - National Library of Spain: CC BY NC SA
- FI1: National Library of Finland: no re-sharing allowed, follow the above section to get the data. (https://digi.kansalliskirjasto.fi/opendata)
- FR1: HIMANIS Project: CC0 (https://www.himanis.org)
- FR2: IMPACT - National Library of France: CC BY NC SA 3.0
- FR3: RECEIPT dataset: CC0 (http://findit.univ-lr.fr)
- NL1: IMPACT - National library of the Netherlands: CC BY
- PL1: IMPACT - National Library of Poland:CC BY
- SL1: IMPACT - Slovak National Library: CC BY NC

Text post-processing such as cleaning and alignment have been applied on the resources mentioned above, so that the Gold Standard and the OCRs provided are not necessarily identical to the originals.


#### Structure:
- **Content** [./lang_type/sub_folder/#.txt]
	- "[OCR_toInput] " => Raw OCRed text to be de-noised.
	- "[OCR_aligned] " => Aligned OCRed text.
	- "[ GS_aligned] " => Aligned Gold Standard text.

The aligned OCRed/GS texts are provided for training and test purposes. The alignment was made at the character level using "@" symbols. "#" symbols correspond to the absence of GS either related to alignment uncertainties or related to unreadable characters in the source document. For a better view of the alignment, make sure to disable the "word wrap" option in your text editor.

The Error Rate and the quality of the alignment vary according to the nature and the state of degradation of the source documents. Periodicals (mostly historical newspapers) for example, due to their complex layout and their original fonts have been reported to be especially challenging. In addition, it should be mentioned that the quality of Gold Standard also varies as the dataset aggregates resources from different projects that have their own annotation procedure, and obviously contains some errors.


#### ICDAR2019 competition:
Information related to the tasks, formats and the evaluation metrics are details on :
https://sites.google.com/view/icdar2019-postcorrectionocr/evaluation


#### References:
 - IMPACT, European Commission's 7th Framework Program, grant agreement 215064
 - Uwe Springmann, Christian Reul, Stefanie Dipper, Johannes Baiter (2018). Ground Truth for training OCR engines on historical documents in German Fraktur and Early Modern Latin.
 - https://digi.nationallibrary.fi , Wiipuri, 31.12.1904, Digital Collections of National Library of Finland 
- EU Horizon 2020 research and innovation programme grant agreement No 770299


#### Contact:
- christophe.rigaud(at)univ-lr.fr
- antoine.doucet(at)univ-lr.fr
- mickael.coustaty(at)univ-lr.fr
- jean-philippe.moreux(at)bnf.fr

L3i - University of la Rochelle, http://l3i.univ-larochelle.fr
BnF - French National Library, http://www.bnf.fr

