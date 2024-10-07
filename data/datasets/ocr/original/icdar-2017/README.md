Corpus for the ICDAR2017 Competition on Post-OCR Text Correction (v1.2 - November 2017)
=> Website: http://l3i.univ-larochelle.fr/ICDAR2017PostOCR
-------------------------------------------------------------------------------

#### Description:
The corpus accounts for 12M OCRed characters along with the corresponding Gold Standard (GS), with an equal share of English- and French-written documents ranging over four centuries. The documents come from different digital collections available, among others, at the National Library of France (BnF) and the British Library (BL). The corresponding GS comes both from BnF's internal projects and external initiatives such as Europeana Newspapers, IMPACT, Project Gutenberg, Perseus and Wikisource.

Versions of the dataset:
- **ICDAR2017_datasetPostOCR_Training_10M.zip** : 83% of the full dataset, provided on demand to train participants' models. 
- **ICDAR2017_datasetPostOCR_Evaluation_2M.zip** : 17% of the full dataset used for the evaluation.
- **ICDAR2017_datasetPostOCR_Full_12M.zip** : full dataset made publicy available after the competition.


#### Licenses: free to use for non-commercial uses, according to sources in details:
- British Library: CC PD (Public Domain)
- French Nationnal Library: Non-commercial use (http://gallica.bnf.fr/html/und/conditions-use-gallicas-contents)
- Wikisource: CC-BY-SA 3.0
- Perseus: CC-BY-SA 3.0
- British Library: CC PD (Public Domain)
- Gutenberg: CC PD (Public Domain)

Post-processing such as cleaning and alignment have been applied on the resources mentioned above, so that the Gold Standard and the OCRs provided are not necessarily identical to the originals.


#### Structure:
- **Metadata** [./metadata.csv]
	- Information: file, date, type, nbAlignedChar

- **Content** [./lang_type/#.txt]
	- 1st line "[OCR_toInput] " => Raw OCRed text to be denoised.
	- 2nd line "[OCR_aligned] " => Aligned OCRed text.
	- 3rd line "[ GS_aligned] " => Aligned Gold Standard.

The aligned OCRed/GS texts are provided for training and test purposes. The alignment was made at the character level using "@" symbols. "#" symbols correspond to the absence of GS either related to alignment uncertainities or related to unreadable characters in the source document. For a better view of the alignment, make sure to disable the "word wrap" option in your text editor.

The Error Rate and the quality of the alignment vary according to the nature and the state of degradation of the source documents. Peridocials (mostly historical newspapers) for example, due to their complex layout and their original fonts have been reported to be especially challenging. In addition, it should be mentioned that the quality of Gold Standard also varies as the dataset agregates ressources from different projects that have their own anotation procedure, and obviously contains some errors.


#### ICDAR2017 competition:
Information related to the tasks, formats and the evaluation metrics are details on :
https://sites.google.com/view/icdar2017-postcorrectionocr/evaluation


#### References:
 - IMPACT, European Commission’s 7th Framework Program, grant agreement 215064
 - EU Competitiveness and Innovation Framework Programme grant ENP 297380


#### Contact:
- guillaume.chiron(at)univ-lr.fr
- antoine.doucet(at)univ-lr.fr
- mickael.coustaty(at)univ-lr.fr
- muriel.visani(at)univ-lr.fr
- jean-philippe.moreux(at)bnf.fr

BnF - French National Library, http://www.bnf.fr
L3i Lab - University of la Rochelle, http://l3i.univ-larochelle.fr
