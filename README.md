# textanalysis

NMF on california DOJ ballots for the past 2 years

:ear:

### What is NMF?

NMF is "straightforwardly" Non-negative matrix factorization. I'll use text analysis / topic modeling as an example to break down the steps:

1. We make a document term matrix. A document term matrix (DTM) represents words and their occurrence in documents in a matrix format. ie

Document 1: ball rim google john
Document 2: ball buzz bux john


document | ball | rim | google | john | bux | buzz |
---------|------|-----|--------|------|-----|-----|
document 1 | 1 |   1 |   1|        1 |  0|  0
document 2| 1 | 1| 0 | 0| 1 | 1


We create a DTM fairly easily using sci-kit learn's CountVectorizer =>  [http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

There's options you can pass through. In `justice.ipynb` I used:

`
vectorizer = text.CountVectorizer(input='filename', stop_words='english', min_df=5, max_df = 0.85)
`

- `stop words` terms that are ignored because there's too many of them. They are defined here: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/stop_words.py
- there's a `lowercase` boolean parameter that you can pass in. It converts all characters to lowercase by default, btw.
- `min_df` = "ignore terms that have a document frequency strictly lower than the given threshold". So in my example, ignore words that appear less than 5 times.
- `max_df`= "ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words)". In my example, ignore anything that appears in the document more than 85%.

So now we can represent our document + terms in vector space a la:
```
[[ 1, 1, 1, 1, 0, 0],
[ 1, 1, 0,0, 1, 1 ]]
(for our initial example)
```

2. Decompose the matrix using matrix factorization. NMF takes the document term matrix and via factorization breaks it down into two new matrices.

- It's the "inverse" of matrix multiplication, ie, what are the two matrices that you can multiply to get the final DTM ?
- A reminder to myself that matrix multiplication is *not* commutative. You can't switch the order of matrix multiplication and get the same thing!

I have problems understanding abstract things, so here's a straightforward example of factorization using numbers:

```
[[ 6,2,4],
[ 9,3,6 ]]
```
is a matrix A of n x m or 2 x 3 (rows by columns)

So in NMF we want to find

A = W x H

where W is a matrix of n x k
and H is a matrix of k x m
and k is less than the rank of matrix A (less than n and m).
The matrices W and H are smaller in size than the original matrix.

It's called non-negative because the resulting matrices are all positive.

In our example, k is concretely the number of topics!
