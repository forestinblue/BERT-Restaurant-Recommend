# KeyBert
## keyword Extract using Korean KeyBERT

### 1. basic KeyBERT


```python
import numpy as np
import itertools
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
```

I will use Korean documents related to food.

```python
doc = """[ ์ ์ฌ ๐์นด๋ฐ๋ ]
-
๐ฝํ๋ ์นด์ธ ๋ (11,000)
๐ฝ๊ฐ๋ผ์ด๊ท๋ (11,500)
๐ฝ์ปต๋๋ค (2,500)
์ ์ฌ์ญ ์ ํฅ๊ฐ์ ์นด๋ฐ๋... ๊ฐ๋ก์๊ธธ ๊ฑด๋ํธ์ ์์นํด ์์ด ์กฐ์ฉํ ์์๋ฌธ ํ๋ ์คใใ
๋ด๋ถ๋ ๋ฐํ์ด๋ธ๋ก ์๋ฆฌ๊ฐ ์ข ๋๋ํ ํธ์ด๊ณ  ํ์ ์ด ๋นจ๋ผ ํ์ฌ๋ ์จ์ดํ์ด ์๋๋ฏ? 
์๊ทธ๋์ฒ ๋ฉ๋ด์ธ ํ๋ ์นด์ธ ๋์ ๋๊น์ค์ ๋ฌ๋ฌํ ํ๋ ์์ค ๊ทธ๋ฆฌ๊ณ  ๊ณ๋์ด๋ถ์ด ๋ฐฅ ์์ ์๋ ๊ตฌ์ฑ์ผ๋ก ๋๊น์ค๋ฌ๋ฒ๋ผ๋ฉด ์ข์ํ  ์ ๋ฐ์ ใ 
์ฌ๊ธฐ์ ๊ฟํ์ ๋ฏธ์์ฅ๊ตญ์ ๋์  ์ปต๋๋ค๋ก ๋ณ๊ฒฝํ๋ฉด 2,500์์ ๊ณ ๊ธฐ ๋ค์ด๊ฐ ๋ผ๋ฉด์ ๋จน์ ์ ์๋ค๋ ๊ฒ!!\n๊ฐ๋ผ์ด๊ท๋์ ๋งค์ฝคํ ๊ท๋์ธ๋ฐ ๋ด๊ฐ ๋ง๋ผ๊ธฐ๋ฆ์ ์ถ๊ฐํด์ ๊ทธ๋ฐ์ง ์ ๋ ์ ์ก๋ณถ์๋ง์ด ใใใใ ๋ง์๋๋ฐ ๋๋ ์ ์ก์ค๋ฌ์์ ๋ค์์ ๋ฐฉ๋ฌธํ๋ฉด ๊ทธ๋ฅ ํ๋ ์นด์ธ ๋ ์ฃผ๋ฌธํ ๋ฏใใ
๋จ๋ฌด์ง๋ ๊ฐ๋ฌด์นจ ์ ์ ๊ณ ์ถ์ง๊ฐ ํ์ด๋ธ๋ง๋ค ์๋๋ฐ ๊ฐ๋ฌด์นจ์ ์งญ์ชผ๋ฆํ๊ฒ ์๊ทผ ๋ณ๋ฏธ!! ์ ์ ๊ณ ์ถ์ง๋ ๋๊น์ค์ ์ฌ๋ ค ๋จน์ผ๋ ์ฐฐ๋ก๐
๋ฒํฐ ์ถ๊ฐํด์ ๋ฐฅ์ ๋น๋ฒผ๋จน๋ ๊ฒ๋ ์์ ๋ง์๊ณ ~
ํฉ๋ฆฌ์ ์ธ ๊ฐ๊ฒฉ์ผ๋ก ์ ์ฌ์ญ ๊ทผ์ฒ์์ ๋ฐฅ ๋จน๊ธฐ ๋๋ฌด ์ข์ ๊ณณ์ด๋ผ ๋ ํซํด์ง ๊ฒ ๊ฐ์ ๋๋!!
๋๊น์ค ์์ฒด๋ ์์ฃผ ํน๋ณํ์ง ์์ง๋ง ๋ํผํ ๋๊น์ค์ ๊ณผํ์ง ์๊ฒ ๋ฌ์ฝค์งญ์งคํ ํ๋ ์์ค ๊ทธ๋ฆฌ๊ณ  ๋ถ๋๋ฌ์ด ๊ณ๋๊ณผ์ ์กฐํ๊ฐ ๋ ๋ ๊ธ๐

๐ํ๋ ์นด์ธ ๋์ ์ผ๋ฐ๊ณผ ์์ด ์๋๋ฐ ์์ ์ฐจ์ด๊ฐ ์๋ ๋ฑ์ฌ๊ณผ ์๋ฑ์ฌ์ ์ฐจ์ด. ๊ฐ๊ฒฉ์ด 4์ฒ์์ด๋ ์ฐจ์ด๊ฐ ๋์ง๋ง ํ์คํ ์์ด ๋ ์ซ๊นํ๋ฉด์ ๋ถ๋๋ฌ์
๐๋ฐฅ ์ถ๊ฐ๋ ๋ฌด๋ฃ๊ธฐ ๋๋ฌธ์ ๋ฒํฐ๋ฅผ 500์์ ์ถ๊ฐํด์ ์๋ฆฌ๋ง๋ค ๋น์น๋ ๊ณ๋ ๊ฐ์ฅ์ ๋ฐฅ์ ๋ฟ๋ ค ๋จน์!
"""
```

Create a document in which only nouns are extracted through a stemmer.
```python
okt = Okt()
tokenized_doc = okt.pos(doc)
tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])
print('ํ์ฌ ํ๊น 10๊ฐ๋ง ์ถ๋ ฅ :',tokenized_doc[:10])
print('๋ช์ฌ ์ถ์ถ :',tokenized_nouns)
```

```python
ํ์ฌ ํ๊น 10๊ฐ๋ง ์ถ๋ ฅ : [('[', 'Punctuation'), ('์ ์ฌ', 'Noun'), ('๐', 'Foreign'), ('์นด๋ฐ', 'Noun'), ('๋', 'Modifier'),
(']', 'Punctuation'),('\n', 'Foreign'), ('-', 'Punctuation'), ('\n', 'Foreign'), ('๐ฝ', 'Foreign')]
๋ช์ฌ ์ถ์ถ : ์ ์ฌ ์นด๋ฐ ๋  ์นด์ธ ๋ ๊ฐ๋ผ์ด ๊ท๋ ์ปต๋ ์ ์ฌ์ญ ์ ํฅ ๊ฐ ์นด๋ฐ ๊ฐ๋ก์๊ธธ ๊ฑด๋ํธ ์์น ์ ์๋ฌธ ์ค ๋ด๋ถ ๋ฐ ํ์ด๋ธ ์๋ฆฌ ์ข ํธ์ด ํ์  ํ์ฌ ์จ์ดํ ์ฒ ๋ฉ๋ด์ธ ๋  ์นด์ธ ๋ 
๋๊น์ค ๋ฌ๋ฌ ๋ ์์ค ๊ณ๋ ์ด๋ถ ๋ฐฅ ์ ๊ตฌ์ฑ ๋๊น์ค ๋ฌ๋ฒ ๋ผ๋ฉด ์ ์ฌ๊ธฐ ๊ฟํ ๋ฏธ์ ์ฅ๊ตญ ๋์  ์ปต๋ ๋ณ๊ฒฝ ๊ณ ๊ธฐ ๋ผ๋ฉด ์ ๊ฒ ๊ฐ๋ผ์ด ๊ท๋ ๋งค์ฝค ๊ท๋ ๋ด ๊ธฐ๋ฆ ์ถ๊ฐ ์ ๋ ๋ณถ์ ๋ง ์ ์ก ๋ค์ 
๋ฐฉ๋ฌธ ๊ทธ๋ฅ ๋  ์นด์ธ ๋ ์ฃผ๋ฌธ ๋จ๋ฌด์ง ๊ฐ ์ ์ ๊ณ ์ถ ํ์ด๋ธ ๊ฐ ์งญ์ชผ๋ฆํ ์๊ทผ ๋ณ๋ฏธ ์ ์ ๊ณ ์ถ ๋๊น์ค ์ฐฐ๋ก ๋ฒํฐ ์ถ๊ฐ ๋ฐฅ ๊ฒ ์์ ํฉ๋ฆฌ ๊ฐ๊ฒฉ ์ ์ฌ์ญ ๊ทผ์ฒ ๋ฐฅ ๋จน๊ธฐ ๊ณณ ๋ ํซ ๊ฒ ๋๋ ๋๊น์ค ์์ฒด 
์์ฃผ ๋ํผ ๋๊น์ค ๊ณผ ๋ฌ์ฝค ๋ ์์ค ๊ณ๋ ์กฐํ ๋  ์นด์ธ ๋ ์ผ๋ฐ ์์ด ์ ์ฐจ์ด ๋ฑ์ฌ ์๋ฑ ์ฌ์ ์ฐจ์ด ๊ฐ๊ฒฉ ์ฐจ์ด ๋ ์์ด ๋ ์ซ๊น ๋ถ๋๋ฌ์ ๋ฐฅ ์ถ๊ฐ ๋ฌด๋ฃ ๊ธฐ ๋๋ฌธ ๋ฒํฐ ์ถ๊ฐ ์๋ฆฌ ๋น์น ๊ณ๋ ๊ฐ์ฅ๋ฐฅ
```





Use sklearn CountVectorizer to extract words.
The reason for using CountVectorizer is, can easily extract n-grams by using the argument of n_gram_range. 
```python
n_gram_range = (2, 3)

count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
candidates = count.get_feature_names()

print('trigram ๊ฐ์ :',len(candidates))
print('trigram ๋ค์ฏ๊ฐ๋ง ์ถ๋ ฅ :',candidates[:5])
```
```python
trigram ๊ฐ์ : 203
trigram ๋ค์ฏ๊ฐ๋ง ์ถ๋ ฅ : ['๊ฐ๊ฒฉ ์ ์ฌ์ญ', '๊ฐ๊ฒฉ ์ ์ฌ์ญ ๊ทผ์ฒ', '๊ฐ๊ฒฉ ์ฐจ์ด', '๊ฐ๊ฒฉ ์ฐจ์ด ์์ด', '๊ฐ๋ผ์ด ๊ท๋']
```


Numericalize keywords through SBERT
```python
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)

top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
print(keywords)
```

### 2. Max Sum Similarity
> [information about Max Sum Similarity](https://aclanthology.org/P08-1007.pdf)

```python
def max_sum_sim(doc_embedding, candidate_embeddings, words, top_n, nr_candidates):
    # ๋ฌธ์์ ๊ฐ ํค์๋๋ค ๊ฐ์ ์ ์ฌ๋
    distances = cosine_similarity(doc_embedding, candidate_embeddings)

    # ๊ฐ ํค์๋๋ค ๊ฐ์ ์ ์ฌ๋
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # ์ฝ์ฌ์ธ ์ ์ฌ๋์ ๊ธฐ๋ฐํ์ฌ ํค์๋๋ค ์ค ์์ top_n๊ฐ์ ๋จ์ด๋ฅผ pick.
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # ๊ฐ ํค์๋๋ค ์ค์์ ๊ฐ์ฅ ๋ ์ ์ฌํ ํค์๋๋ค๊ฐ์ ์กฐํฉ์ ๊ณ์ฐ
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]
```

A relatively high number of nr_candidates creates a variety of keywords.
```python
top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
print(keywords)
```
```python
['์ ๋ ๋ณถ์ ์ ์ก', '๋ ์์ค ๊ณ๋ ์ด๋ถ', 'ํ์ด๋ธ ์งญ์ชผ๋ฆํ ์๊ทผ', '๋ฌ๋ฌ ๋ ์์ค ๊ณ๋', '๋ฌ์ฝค ๋ ์์ค ๊ณ๋']
```
