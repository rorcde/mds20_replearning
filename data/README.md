# Data Description

There are three main Datasets used in this work, **``Movie Rewiew``**, **``Book Corpus``** and **``Money tracsactions``**. you can find more details about each dataset below.
## 1_ Movie review

tabel below will give you summery of all sub-datasets which will be incoluded in **Movie Review dataset**.

The link to Movie Review dataset is [here](http://www.cs.cornell.edu/people/pabo/movie-review-data/).

`The dataset can be downloaded by running notebooks/raw_data.ipynb`


<img src="https://raw.githubusercontent.com/rodrigorivera/mds20_replearning/master/data/imgs/movie_review%20dataset.png?token=AH237P4X67XDB3HIRWJFBJS73ZF32" width='600'>

### 1.1_ Polarity_html :

* The original (unprocessed, unlabeled) source files from which the processed, labeled, and (randomly) selected data(review_polarity) was derived. 


### 1.2_ review_polarity :

* txt_sentoken:  contains 2000 texts.

    * 1000 positive texts.
    * 1000 negative texts.
    * each line is a single sentence.
    
### 1.3_ rt-polaritydata :

* rt-polarity.pos: contains 5331 positive snippets.
* rt-polarity.neg: contains 5331 negative snippets.
* each line is a single sentence.
* drived from: subjectivity/subjective folder.
* Label Decision: in parent .html file, those revirwes which marked with  ``fresh`` are positiveand the one with ``rotten`` are negative.


### 1.4_ scale_whole_review : 

* raw revirws: before passing through tokenization, sentence separation, and subjectivity extraction.
* there are four reviewers sub-directories which each has:
    * id.txt: each line is one paragraph of the review

### 1.5_ scale_data :

* there are four author sub-directories and at each:
    * id : file id’s in polarity_html
    * label.3class : for the {0,1,2} three-category classification task
    * label.4class : for the {0,1,2,3} four-category classification task
    * rating(normalized ratings) : [0-1] with step size 0.1
    * subj : review text
    
    
### 1.6_ subjectivity_html :

* texts in ``rotten_imdb`` is extracted from this folder.
    
* obj/(2002,2003) : html files (all sentences from IMDb plot summaries are labeled as objective).
* subj/2002 :plot summaries (all snippets from the RottenTomatoes.com are labeled as subjective).


### 1.7_ rotten_imdb : 
* quote.tok.gt9.5000 : 5000 subjective sentences.
* plot.tok.gt9.5000 : 5000 objective sentences.
* each line is a single sentence (at least 10 tokens.)


## 2_ Book Corpus dataset:

The Book Corpus dataset containes about more than 17800 books from various categories. the dataset which we used in our worde combined all these books in two `.txt` file namely `books_large_p1` and `books_large_p2`.

`The dataset can be downloaded by running data/notebooks/load_book_corpus_paper.ipynb`

### tabel below is statistical review from this dataset.
* in original [paper](https://arxiv.org/abs/1506.06726) the numbers are slightly different, the reason for this differences is that in main paper numbers, points `.`  at the end of sentences and some semi-words like `half-ling` considered as a word however we used standard pytorch tokenizer which doesn't counts them as a token. 

<img src="https://raw.githubusercontent.com/rodrigorivera/mds20_replearning/master/data/imgs/book_corpus.png?token=AH237P4NBSTKPZXNLTMSI2C73ZGAY" width='600'>

* tabel is derived from `data/notebooks/book_corpus_description.ipynb`


## 3_ Money tracsactions dataset:

`The description can be found in data/notebooks/Money_transaction_description.ipynb`


* the dataset can be downloaded from this [Link](https://www.kaggle.com/c/python-and-analyze-data-final-project)
* it contains four csv file as follows:
    * gender_train.csv [Link](https://www.dropbox.com/s/z8hu28rcd7gxpz8/gender_train.csv)
    * tr_mcc_codes.csv [Link](https://www.dropbox.com/s/23k2l72ko5g5f4n/tr_mcc_codes.csv)
    * tr_types.csv [Link](https://www.dropbox.com/s/3d7qnpq3ckbajh1/tr_types.csv)
    * transactions.csv [Link](https://www.dropbox.com/s/h5vqzdnfdmuqwh9/transactions.csv)
    
* Dataset:

<img src="https://raw.githubusercontent.com/rodrigorivera/mds20_replearning/master/data/imgs/money_transaction_tabel.png?token=AMWMJ6J4QE7SGAMULBQLJE27ZEUCY" width='500'>

<img src="https://raw.githubusercontent.com/rodrigorivera/mds20_replearning/master/data/imgs/pop_catagories_fig.png?token=AMWMJ6M4U3MHK3LCAE2JBT27ZEUN4" width='500'>
