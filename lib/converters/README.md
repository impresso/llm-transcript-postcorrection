## 

## Converters

The data format in `jsonl` is:
```
{
Const.FILE: the file path of the concerned text,
Const.OCR: {Const.LINE: the output of an OCR text of a line if exists otherwise None,
            Const.SENTENCE: the output of an OCR text of a sentence (that contains the line),
            Const.REGION: the output of an OCR of the full text},
Const.GROUND: {Const.LINE: the groundtruth text line,
               Const.SENTENCE: the grountruth of a sentence,
               Const.REGION: the grountruth full text}}

```

All converters have the same parameters (`input_dir` and `output_dir`) and generate a dataset `$DATASET$` in `$OUTPUT_DIR` (e.g., `data/ocr/converted`).


## Makefile converters

For running all converters:
`./convert.sh /path/to/your/data`

Example:
`./convert.sh ../../data/datasets`



### Converter ICDAR 2017 & 2019
The dataset contains text files [./lang_type/#.txt]
	- 1st line "[OCR_toInput] " => Raw OCRed text to be denoised.
	- 2nd line "[OCR_aligned] " => Aligned OCRed text.
	- 3rd line "[ GS_aligned] " => Aligned Gold Standard.
It also contains eval_metadata.csv with years and type of publication that are added to the converted json lines.
```
python icdar_converter.py --input_dir ../../data/datasets/ocr/original/icdar-2017/ \
                          --output_dir ../../data/datasets/ocr/converted
```

### Converter impresso-nzz
This NZZ ground truth contains several directories:
   - `xml`: a folder containing all the XML files (ABBYY output as well as ground truth)
   - `ABBYY_FineReader_XIX`: The original OCRed material from 2005 provided by the NZZ in the form of PDF files. The images are extracted as well as the text with TET PDFLib an saved them as .tiff files and .tetml files, respectively. The newspaper pages were OCR-ised with ABBYY FineReader XIX, an older product from ABBYY for gothic letter. 
   - `ABBYY_FineReader_Server11`: A version produced from Transkribus-internal (see below) ABBYY FineReader Engine 11.
   - `NZZ_groundtruth`: the manually corrected texts from the 167 pages of the NZZ.
   - the ground truth contains 304,268 words and 43,151 lines.
    
The converter maps the files in both folders (`NZZ_groundtruth` to `ABBYY_FineReader_Server11`), along with the `region_id` and `line_id` in every region. If the OCRed region or text line is not found, it is disconsired. We could not map `ABBYY_FineReader_XIX`.

```
python nzz_converter.py --input_dir ../../data/datasets/ocr/original/icdar-2017/ \
                          --output_dir ../../data/datasets/ocr/converted
```

### Converter Overproof
Per-year files containing the line-by-line original OCR and human correction line pairs for these 30,301 articles are available here. Each file contains multiple articles. Each article starts with a header line identified as starting with *$*OVERPROOF*$* , and followed by original OCR text and current version text pairs, for example, the 1842 file starts like this:

*$*OVERPROOF*$* 12409236 year 1842 type Article title The Sydney Morn
lUHSr.1 rÃÂ«10.vl TUS IMXHRÃÂOR.||@@||NEWS FROM THE INTERIOR.
(From our various Correspondents.)||@@||(From our various Correspondents.)
NEWCASTLE.||@@||NEWCASTLE.
A vnnY melancholy accident, occa ioncd by||@@||A very melancholy accident, occasioned by
the incautious use of fim-arini, ocuiirrcd here||@@||the incautious use of fire-arms, occurred here
on Saturday last, to one ot the privates of the||@@||on Saturday last, to one of the privates of the

```
python overproof_converter.py --input_dir ../../data/datasets/ocr/original/overproof/ \
                          --output_dir ../../data/datasets/ocr/converted
```

### Converter Quaero
This package contains data and tools developed within the QUAERO project about Extended Named Entities in Broadcast News data. The folder contains:

- test/ the test data used in the Quaero Broadcast News evaluation,
  described in Galibert et al., IJCNLP 2011.

  - `ref-raw-annot/` txt files, manually annotated

  - `ref-norm-annot/` txt files, manually annotated and automatically
    normalized using ne-normalize-generales

  - `alignements/` the manual transcription files aligned on signal
    times (see Galibert et al., IJCNLP 2011 for explanations)

  - `asr-annot/` the automatically built annotation reference projected
    on the different ASR outputs (see Galibert et al., IJCNLP 2011 for
    explanations)

  - `output-system-limsi/` the hypotheses produced by the LIMSI system
    described in Dinarelli and Rosset, IJCNLP 2011.
    
 We only work with `ref-norm-annot/`. 
 
 The data format consists in textlines annotated with entities that were corrected (uppercased).
 ```
je suis content d' organiser des retrouvailles comme celles d' <pers.ind> <name.first> Isabelle </name.first> <name.last> Alonzo </name.last> </pers.ind> et <pers.ind> <name.first> Olivier </name.first> <name.last> de Kersauson </name.last> </pers.ind> , qui ne s' Ètaient pas vus depuis , &oh-la-la depuis
depuis une Èpoque %% &oh-la du <pers.ind> <title> chevalier </title> <name.last> Bayard </name.last> </pers.ind> , en quelque sorte
longtemps
contente de revoir <pers.ind> <name.first> Olivier </name.first> </pers.ind> ?
mais je suis ravie , Áa fait longtemps que je l' avais pas vu , il change pas , il a toujours tous ses cheveux %% *c' est
je ne dis rien , je ne dis rien
t' avancer sur le physique
je ne m' avance pas
```

To discuss here what is the ground truth.

```
python quaero_converter.py --input_dir ../../data/datasets/ocr/original/quaero-broadcast/ \
                          --output_dir ../../data/datasets/ocr/converted
```

