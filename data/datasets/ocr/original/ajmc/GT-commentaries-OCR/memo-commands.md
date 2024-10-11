# Commands to slice and dice

Schneidewin: 

```bash

cd /Users/matteo/Documents/AjaxMultiCommentary/GT-commentaries-OCR

python ~/Documents/Lace2-tools/training_set_from_tsv.py --csvFile sophokle1v3soph/sophokle1v3soph_training.tsv --imageDir /Volumes/GoogleDrive/.shortcut-targets-by-id/12W80v9RNL9hQvtqnATpvd2zdMvuySsn4/AjaxMultiCommentary/data/commentary_data/schneidewin/images/sophokle1v3soph-images/ --outputDir sophokle1v3soph/GT-pairs

cat sophokle1v3soph/pairs/*.txt | python ~/Documents/Lace2-tools/unique_chars.py > sophokle1v3soph/unique_chars.log
```

Campbell:

```bash
python python ~/Documents/Lace2-tools/training_set_from_tsv.py --csvFile cu31924087948174/cu31924087948174_training.tsv --imageDir /Volumes/GoogleDrive/.shortcut-targets-by-id/12W80v9RNL9hQvtqnATpvd2zdMvuySsn4/AjaxMultiCommentary/data/commentary_data/campbell/images/cu31924087948174-images/ --outputDir cu31924087948174/pairs/
```

Jebb:

```bash
python ~/Documents/Lace2-tools/training_set_from_tsv.py --csvFile sophoclesplaysa05campgoog/sophoclesplaysa05campgoog_training.tsv --imageDir /Volumes/GoogleDrive/.shortcut-targets-by-id/12W80v9RNL9hQvtqnATpvd2zdMvuySsn4/AjaxMultiCommentary/data/commentary_data/jebb/images/sophoclesplaysa05campgoog_png/ --outputDir sophoclesplaysa05campgoog/GT-pairs/
```

```bash
python ~/Documents/Lace2-tools/training_set_from_tsv.py --csvFile sophoclesplaysa05campgoog/sophoclesplaysa05campgoog_training_FROM-SVEN-LOCAL.tsv --imageDir /Volumes/GoogleDrive/.shortcut-targets-by-id/12W80v9RNL9hQvtqnATpvd2zdMvuySsn4/AjaxMultiCommentary/data/commentary_data/jebb/images/sophoclesplaysa05campgoog_png/ --outputDir sophoclesplaysa05campgoog/GT-pairs/
```

Lobeck:

```bash
python ~/Documents/Lace2-tools/training_set_from_tsv.py --csvFile bsb10234118/bsb10234118_training.tsv --imageDir /Volumes/GoogleDrive/.shortcut-targets-by-id/12W80v9RNL9hQvtqnATpvd2zdMvuySsn4/AjaxMultiCommentary/data/commentary_data/lobeck/images/bsb10234118-images/ --outputDir bsb10234118/pairs/
```

Wecklein:


```bash
python ~/Documents/Lace2-tools/training_set_from_tsv.py --csvFile Wecklein1894/Wecklein1894_training.tsv --imageDir /Volumes/GoogleDrive/.shortcut-targets-by-id/12W80v9RNL9hQvtqnATpvd2zdMvuySsn4/AjaxMultiCommentary/data/commentary_data/wecklein/images/Wecklein1894-images/ --outputDir Wecklein1894/pairs/
```

# Commands to copy images

Wecklein:

```bash
cat Wecklein1894/Wecklein1894_training.tsv | cut -f 1 | uniq | while read i; do cp /Volumes/GoogleDrive/.shortcut-targets-by-id/12W80v9RNL9hQvtqnATpvd2zdMvuySsn4/AjaxMultiCommentary/data/commentary_data/wecklein/images/Wecklein1894-images/$i Wecklein1894/imgs/ ; echo $i ;done
```

Jebb:

```bash
cat sophoclesplaysa05campgoog/sophoclesplaysa05campgoog_training.tsv | cut -f 1 | uniq | while read i; do cp /Volumes/GoogleDrive/.shortcut-targets-by-id/12W80v9RNL9hQvtqnATpvd2zdMvuySsn4/AjaxMultiCommentary/data/commentary_data/jebb/images/sophoclesplaysa05campgoog_png/$i sophoclesplaysa05campgoog/imgs/ ; done
```

Schneidewin: 

```bash
cat sophokle1v3soph_training.tsv | cut -f 1 | uniq | while read i; do cp /Volumes/GoogleDrive/.shortcut-targets-by-id/12W80v9RNL9hQvtqnATpvd2zdMvuySsn4/AjaxMultiCommentary/data/commentary_data/schneidewin/images/sophokle1v3soph-images/$i . ; done
```

Lobeck:

```bash
cat bsb10234118/bsb10234118_training.tsv | cut -f 1 | uniq | while read i; do cp /Volumes/GoogleDrive/.shortcut-targets-by-id/12W80v9RNL9hQvtqnATpvd2zdMvuySsn4/AjaxMultiCommentary/data/commentary_data/lobeck/images/bsb10234118-images/$i bsb10234118/imgs/ ; echo $i ;done
```

Campbell:

```bash
cat cu31924087948174/cu31924087948174_training.tsv | cut -f 1 | uniq | while read i; do cp /Volumes/GoogleDrive/.shortcut-targets-by-id/12W80v9RNL9hQvtqnATpvd2zdMvuySsn4/AjaxMultiCommentary/data/commentary_data/campbell/images/cu31924087948174-images/$i cu31924087948174/imgs/ ; echo $i ;done
```

# Commands to count GT lines in evaluation set

```bash
egrep -i '(0012|0017|0085|0081|0125|0048|0049|0140)' sophokle1v3soph_training.tsv | wc -l
```


# Unicode NFC conversion with `uconv`

```bash
for f in AjaxMultiCommentary/GT-commentaries-OCR/data/Wecklein1894/GT-pairs/*.txt; do cat $f | uconv -f utf8 -t utf8 -x nfc > $f ; done

cat data/bsb10234118/GT-pairs/*.txt | python ~/Documents/Lace2-tools/unique_chars.py > data/bsb10234118/unique_chars_nfc.log
```

# Commands to copy OLR files only for GT images

Campbell:

```bash
egrep -i '(056.png|0120.png|0132.png|036.png|040.png|014.png|010.png|063.png|055.png|064.png|087.png|0107.png|0118.png|0142.png|034.png|037.png|016.png|028.png|050.png|054.png|069.png|075.png|086.png|097.png|041.png|035.png|023.png|022.png|045.png|0113.png|0127.png|0130.png|0141.png|0148.png|038.png|039.png|012.png|011.png|0102.png|083.png)' ~/Google\ Drive/My\ Drive/AjaxMultiCommentary/data/commentary_data/cu31924087948174/olr/annotations.csv > cu31924087948174/cu31924087948174_olr.tsv
```
Jebb:

```bash
# jebb
egrep -i '(0177.png|099.png|0112.png|0146.png|094.png|088.png|0336.png|050.png|014.png|012.png|0101.png|0117.png|0106.png|0118.png|0132.png|089.png|0346.png|018.png|017.png|0331.png|0135.png|095.png|0169.png|0166.png|0162.png|0104.png|090.png|0337.png|020.png|022.png|0326.png|0149.png|0157.png|0180.png|0192.png|0204.png|087.png|0340.png|0315.png|019.png|035.png|0136.png|0137.png)' ~/Google\ Drive/My\ Drive/AjaxMultiCommentary/data/commentary_data/sophoclesplaysa05campgoog/olr/annotations.csv > sophoclesplaysa05campgoog/sophoclesplaysa05campgoog_olr.tsv
```

Lobeck:

```bash
egrep -i '(bsb10234118_0027.png|bsb10234118_0024.png|bsb10234118_0519.png|bsb10234118_0116.png|bsb10234118_0090.png|bsb10234118_0042.png|bsb10234118_0117.png|bsb10234118_0017.png|bsb10234118_0023.png|bsb10234118_0501.png|bsb10234118_0010.png|bsb10234118_0043.png|bsb10234118_0019.png|bsb10234118_0050.png|bsb10234118_0011.png|bsb10234118_0013.png|bsb10234118_0012.png|bsb10234118_0096.png|bsb10234118_0067.png|bsb10234118_0015.png|bsb10234118_0014.png|bsb10234118_0029.png|bsb10234118_0091.png|bsb10234118_0009.png|bsb10234118_0170.png)' ~/Google\ Drive/My\ Drive/AjaxMultiCommentary/data/commentary_data/bsb10234118/olr/annotations.csv > bsb10234118/bsb10234118_olr.tsv
```

Schneidewin:

```bash
egrep -i '(sophokle1v3soph_0011.png|sophokle1v3soph_0070.png|sophokle1v3soph_0016.png|sophokle1v3soph_0125.png|sophokle1v3soph_0048.png|sophokle1v3soph_0012.png|sophokle1v3soph_0120.png|sophokle1v3soph_0015.png|sophokle1v3soph_0085.png|sophokle1v3soph_0081.png|sophokle1v3soph_0013.png|sophokle1v3soph_0095.png|sophokle1v3soph_0007.png|sophokle1v3soph_0115.png|sophokle1v3soph_0009.png|sophokle1v3soph_0047.png|sophokle1v3soph_0010.png|sophokle1v3soph_0073.png|sophokle1v3soph_0074.png|sophokle1v3soph_0140.png|sophokle1v3soph_0017.png)' ~/Google\ Drive/My\ Drive/AjaxMultiCommentary/data/commentary_data/sophokle1v3soph/olr/annotations.csv > sophokle1v3soph/sophokle1v3soph_olr.tsv
```

Wecklein:

```bash
egrep -i '(Wecklein1894_0007.png|Wecklein1894_0109.png|Wecklein1894_0080.png|Wecklein1894_0087.png|Wecklein1894_0016.png)' ~/Google\ Drive/My\ Drive/AjaxMultiCommentary/data/commentary_data/Wecklein1894/olr/annotations.csv > Wecklein1894/Wecklein1894_olr.tsv
```