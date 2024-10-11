# Error Correcting HTR’ed Byzantine Text
This is the repository of the [HTREC challenge](https://www.aicrowd.com/challenges/htrec-2022). 

The 1st version of this repository was kindly developed by [@Connalia](https://github.com/Connalia/). The HTR model, the train and the original test data were developed by [@vivianpl](https://github.com/vivianpl). The synthetic test data along with unlimited help with the organisation are attributed to [@vasilikikou](https://github.com/vasilikikou).

## Dataset

The data folder comprises the dataset in the form of `train.csv` and `test.csv`, the latter including data from the `original_test.csv` and the `synthetic_test.csv`

## Code

* An exploratory data analysis appears in `eda.ipynb`
* The rule-based approach exists in `rbs.ipynb`, also described with pseudocode in `rbs_pseudocode.pdf`
* The ByT5 approach exists in `neural.ipynb`

## Citation

If you find our work useful to your research, please cite this work as:

```
@inproceedings{pavlopoulos-2023,
    title = "Error Correcting HTR’ed Byzantine Text",
    author = "John Pavlopoulos, Vasiliki Kougia, Paraskevi Platanou, Stepan Shabalin, Konstantina Liagkou, Emmanouil Papadatos, Holger Essler, Jean-Baptiste Camps, and Franz Fischer",
    year = "2023",
}
```
