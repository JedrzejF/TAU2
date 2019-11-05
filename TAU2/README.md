WMT2017 Czech-English machine translation challenge for news
=============================================================

Translate news articles from Czech into English.

This is WMT2017 news challenge reformatted as a Gonito.net challenge,
all the data were taken from <http://www.statmt.org/wmt17/translation-task.html>.

BLEU is used as the evaluation metric.

Directory structure
-------------------

* `README.md` — this file
* `config.txt` — configuration file
* `train/` — directory with training data
* `train/commoncrawl.tsv.xz` — Common Crawl parallel corpus
* `train/news-commentary-v12.tsv.xz` — News Commentary parallel corpus
* `train/europarl-v7.tsv.xz` — Europarliament parallel corpus
* `dev-0/` — directory with dev (test) data (Newstest 2013)
* `dev-0/in.tsv` — German input text for the dev set
* `dev-0/expected.tsv` — English reference translation for the dev set
* `test-A` — directory with test data
* `test-A/in.tsv` — German input data for the test set (WMT2017 test set)
* `test-A/expected.tsv` — English reference translation for the test set

Training sets
-------------

All training sets were compressed with xz, use `xzcat` to decompress:

    $ xzcat train/*.tsv.xz | ...

The pairs where German or English side is empty were removed from the
training sets.

Test sets
---------

Reference English translations in the dev and test sets is not tokenised.

Monolingual data
----------------

Monolingual data was not included here.
