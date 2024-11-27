# Ground truth for Neue ZÃ¼rcher Zeitung black letter period
In order to be able to assess the OCR quality of newspapers and also in order to be able to train new OCR recognition models, it is necessary to have a ground truth at one's disposal.

## Sampling
The Neue ZÃ¼rcher Zeitung (NZZ) has been publishing in black letter from its very first issue in 1780 until 1947. From this time period, we randomly sampled one frontpage per year, resulting in a total of 167 pages. We chose frontpages because they typically contain highly relevant material and because we want to make sure not to sample pages containing exclusively advertisements or stock information. During certain periods, the NZZ was published several times a day, and there were supplements, too. Due to incomplete metadata, the sampling included frontpages from supplements.

## Ground truth production
In order to speed up the process of the ground truth production, we uploaded the 167 images to [Transkribus](https://transkribus.eu/Transkribus) and OCRed the text with its internal ABBYY FineReader Server 11. We then used Transkribus to manually correct the text. For about 100 pages, we corrected on the word level and the line level was automatically synchronized by Transkribus. For the rest of the pages, only the line level was corrected.

When the transcription of 120 pages was finished, the Transkribus team trained an HTR model with which we recognized the text from the remaining pages. This significantly speeded up our process, however, these XML files do not contain any word-level information.

### Guidelines

 - *punctuation*
   - the old writing styles use the equality sign "=" to split words over two lines. We transcribed this as "-", like we would use it today.
   - special symbols were only rarely transcribed (sometimes the newspaper used a symbol like a triangle, a dagger or a cross to mark a specific canton or author).

 - *spelling*
   - in black letter, there is no distinction between capital "I" and capital "J". We decided to transcribe the words starting with either letter in the way current spelling rules would require it, e.g. we would write "Jakob" instead of "Iakob", and "Insekt" instead of "Jnsekt".
   - the ligature "sz" is kept and transcribed as "ÃŸ"
   - spaced letters (Sperrschrift) are fused (using the merge tool of transkribus for the word-level transcriptions)
   - sometimes hyphenated words contain spaces, e.g. "Landboten - Korrespondent". We adopted the current spelling "Landboten-Korrespondent" for such cases.
   -  "NÂ°" is transcribed as "Nr.".
   - Long "s" is transcribed as normal "s".

 - *regions*
   - in Transkribus, it is possible to either transcribe word-based or line-based. Generally, our ground truth is line-based. We have a line-based transcription for all pages. For the following pages, we also provide word-based transcriptions: **Word-based transcription for years:**
1780-1895, 1898, 1900, 1905, 1908, 1910, 1913, 1915, 1918, 1920, 1923, 1925, 1930, 1933, 1935, 1938, 1940, 1943, 1945, 1946
TOTAL: 134 years --> the word boxes for these years should also be right.
   - we corrected all baselines, so the ground truth can be used to train HTR models in Transkribus
   - moreover, we corrected the text regions where necessary
   - for the pages where we detected the text with the HTR model, we straightened the line boxes
 
 ## Known issues and additional remarks
There are some remaining inconsistencies in the transcription of quotation marks. In some cases, we used the curly quotation marks for quotation marks on the baseline, whereas in others, the straight quotation marks were used to mark the end of quotation.

There are pages which have been slightly cut at the right-hand side. This stems from the digitisation process by the NZZ. 

<b><span style="color: red">Please note that for pages which have only been corrected on the line level, the ground truth XML files still contains the uncorrected text on the word level!</span></b>
 
 ## Training and test splits used for Transkribus HTR model evaluation
 Our DH2019 paper about Transkribus HTR for improving the OCR of black letter in newspaper texts used the following years for testing:
 1780, 1790, 1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1904, 1910, 1915, 1929, 1939
 The repository contains a text file with the exact list of names.
 
 ## Content
This NZZ ground truth contains several directories:
 - `xml`: a folder containing all the XML files (ABBYY output as well as ground truth)
   - `ABBYY_FineReader_XIX`: The original OCRed material from 2005 provided by the NZZ in the form of PDF files. We extracted the images as well as the text with TET PDFLib an saved them as .tif files and .tetml files, respectively. The newspaper pages were OCR-ised with ABBYY FineReader XIX, an older product from ABBYY for gothic letter. 
   - `ABBYY_FineReader_Server11`: A version produced from Transkribus-internal (see below) ABBYY FineReader Engine 11.
   - `NZZ_groundtruth`: the manually corrected texts from the 167 pages of the NZZ.
    - the ground truth contains 304,268 words and 43,151 lines.
 - `img`: The images we extracted from the PDF files provided by the NZZ.

 
## Transcribers:
 - Isabelle Meraner
 - Camille Watter
 - Simon Clematide siclemat@cl.uzh.ch
 - Phillip StrÃ¶bel pstroebel@cl.uzh.ch - in case of questions
 
 Feel free to open issues if you've encountered a problem or any difficulties while working with this data set. Questions are also welcome.

## Final remarks
All the data, which includes .xml and image files, in this repository is licensed under a Creative Commons license as specified in the file LICENSE.txt. This ground truth can be used for academic purposes. 

Neue ZÃ¼rcher Zeitung black letter ground truth (c) by Phillip StrÃ¶bel and Simon Clematide

Neue ZÃ¼rcher Zeitung black letter ground truth is licensed under a
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt>.

<b>If you use it, please indicate the source as</b>
```
@inproceedings{clematide-stroebel-2019,
  author = "StrÃ¶bel, Phillip and Clematide, Simon",
  title = "Improving OCR of Black Letter in Historical Newspapers: The Unreasonable Effectiveness of HTR Models on Low-Resolution Images",
  year = 2019,
  booktitle = "Proceedings of the Digital Humanities 2019, (DH2019)",
  note = "accepted"
}
```
