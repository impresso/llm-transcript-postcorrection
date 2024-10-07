# OCR Ground Truth for Historical Commentaries

[![DOI](https://zenodo.org/badge/365153616.svg)](https://zenodo.org/badge/latestdoi/365153616)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

The dataset OCR ground truth for historical commentaries (GT4HistComment) was created from the public domain subset of scholarly commentaries on Sophocles' *Ajax*. Its main goal is to enable the evaluation of the OCR quality on printed materials that contain a mix of Latin and polytonic Greek scripts. It consists of five 19C commentaries written in German, English, and Latin, for a total of 3,356 GT lines.

## Data

GT4HistComment are contained in `data/`, where each sub-folder corresponds to a different publication (i.e. commentary). For each each commentary we provide the following data:
- `<commentary_id>/GT-pairs`: pairs of image/text files for each GT line
- `<commentary_id>/imgs`: original images on which the OCR was performed
- `<commentary_id>/<commentary_id>_olr.tsv`: OLR annotations with image region coordinates and layout type ground truth label

The OCR output produced by the Kraken + Ciaconna pipeline was manually corrected by a pool of annotators using the [Lace platform](https://github.com/brobertson/Lace2/). In order to ensure the quality of the ground truth datasets, an additional verification of all transcriptions made in Lace was carried out by an annotator on line-by-line pairs of image and corresponding text.



## Commentary overview


|        ID     | Commentator     | Year | Languages | Image source | Line example |
|---------------|-----------------|------|-----------|--------------|--------------|
| [bsb10234118](./data/bsb10234118/)   | Lobeck [1]      | 1835 | Greek, Latin |[BSB](http://mdz-nbn-resolving.de/urn:nbn:de:bvb:12-bsb10234118-7)      | ![](data/bsb10234118/GT-pairs/bsb10234118_0096_28.png)|
|[sophokle1v3soph](data/sophokle1v3soph)| Schneidewin [2] | 1853  | Greek, German | [Internet Archive](https://archive.org/details/sophokle1v3soph/page/n49/mode/2up) | ![](data/sophokle1v3soph/GT-pairs/sophokle1v3soph_0140_30.png)|
| [cu31924087948174](./data/cu31924087948174/) | Campbell [3]    | 1881  | Greek, English | [Internet Archive]( https://archive.org/details/cu31924087948174) | ![](data/cu31924087948174/GT-pairs/cu31924087948174_0063_70.png) |
| [sophoclesplaysa05campgoog](./data/sophoclesplaysa05campgoog/) |Jebb [4] | 1896  | Greek, English | [Internet Archive](https://archive.org/details/sophoclesplaysa05campgoog) | ![](data/sophoclesplaysa05campgoog/GT-pairs/sophoclesplaysa05campgoog_0136_55.png) |
| [Wecklein1894](data/Wecklein1894/)  | Wecklein [5]  | 1894 [5] | Greek. German | internal | ![](data/Wecklein1894/GT-pairs/Wecklein1894_0087_6.png) |  

## Stats

Line, word and char counts for each commentary are indicated in the following table. Detailled counts for each region can be found [here](https://docs.google.com/spreadsheets/d/1BxtB38WbB1fFplp5mVncfHPH77Z5Z3dZT_0akFTxr0E/edit?usp=sharing).

| ID  | Commentator | Type | lines | words | all chars | greek chars|
|-------------|-------------|---------------------------|----------------|------|-------|-------|
| bsb10234118 | Lobeck | training |   574   | 2943  | 16081 | 5344  |
| bsb10234118 | Lobeck | groundtruth | 202   | 1491  | 7917  | 2786  |
| sophokle1v3soph | Schneidewin | training |   583   | 2970  | 16112 | 3269  |
| sophokle1v3soph | Schneidewin |   groundtruth        |  382   | 1599  | 8436  | 2191  |
| cu31924087948174 | Campbell    | groundtruth          | 464   | 2987  | 14291 | 3566  |
| sophoclesplaysa05campgoog| Jebb        | training |   561   | 4102  | 19141 | 5314  |
| sophoclesplaysa05campgoog | Jebb | groundtruth | 324   | 2418  | 10986 | 2805  |
| Wecklein1894 | Wecklein | groundtruth |  211   | 1912  | 9556  | 3268  |

### Commentary editions used:

- [1] Lobeck, Christian August. 1835. *Sophoclis Aiax*. Leipzig: Weidmann.
- [2] Sophokles. 1853. *Sophokles Erklaert von F. W. Schneidewin*. Erstes Baendchen: Aias. Philoktetes. Edited by Friedrich Wilhelm Schneidewin. Leipzig: Weidmann.
- [3] Lewis Campbell. 1881. *Sophocles*. Oxford : Clarendon Press.
- [4] Wecklein, Nikolaus. 1894. *Sophokleus Aias*. München: Lindauer.
- [5] Jebb, Richard Claverhouse. 1896. *Sophocles: The Plays and Fragments*. London: Cambridge University Press.

## Citation

If you use this dataset in your research, please cite the following publication:

```latex
@inproceedings{romanello_optical_2021,
  title = {Optical {{Character Recognition}} of 19th {{Century Classical Commentaries}}: The {{Current State}} of {{Affairs}}},
  booktitle = {The 6th {{International Workshop}} on {{Historical Document Imaging}} and {{Processing}} ({{HIP}} '21)},
  author = {Romanello, Matteo and Sven, Najem-Meyer and Robertson, Bruce},
  year = {2021},
  publisher = {{Association for Computing Machinery}},
  address = {{Lausanne}},
  doi = {10.1145/3476887.3476911}
}
```

## Acknowledgements

Data in this repository were produced in the context of the Ajax Multi-Commentary project, funded by the Swiss National Science Foundation under an Ambizione grant [PZ00P1\_186033](http://p3.snf.ch/project-186033).

Contributors: Carla Amaya (UNIL), Sven Najem-Meyer (EPFL), Matteo Romanello (UNIL), Bruce Robertson (Mount Allison University).
