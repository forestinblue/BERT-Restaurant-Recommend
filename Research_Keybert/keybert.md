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
doc = """[ 신사 📍카바동 ]
-
🍽타레카츠동 (11,000)
🍽가라이규동 (11,500)
🍽컵누들 (2,500)
신사역 신흥강자 카바동... 가로수길 건너편에 위치해 있어 조용히 입소문 타는 중ㅎㅎ
내부는 바테이블로 자리가 좀 넉넉한 편이고 회전이 빨라 현재는 웨이팅이 없는듯? 
시그니처 메뉴인 타레카츠동은 돈까스에 달달한 타레소스 그리고 계란이불이 밥 위에 있는 구성으로 돈까스러버라면 좋아할 수 밖에 ㅠ
여기서 꿀팁은 미소장국을 닛신 컵누들로 변경하면 2,500원에 고기 들어간 라면을 먹을 수 있다는 것!!\n가라이규동은 매콤한 규동인데 내가 마라기름을 추가해서 그런지 유독 제육볶음맛이 ㅋㅋㅋㅋ 맛있는데 넘나 제육스러워서 다음에 방문하면 그냥 타레카츠동 주문할듯ㅎㅋ
단무지랑 갓무침 유자 고추지가 테이블마다 있는데 갓무침은 짭쪼름한게 은근 별미!! 유자 고추지는 돈까스에 올려 먹으니 찰떡😋
버터 추가해서 밥에 비벼먹는 것도 왕왕 맛있고~
합리적인 가격으로 신사역 근처에서 밥 먹기 너무 좋은 곳이라 더 핫해질 것 같은 느낌!!
돈까스 자체는 아주 특별하지 않지만 두툼한 돈까스와 과하지 않게 달콤짭짤한 타레소스 그리고 부드러운 계란과의 조화가 냠냠긋😙

👉타레카츠동은 일반과 상이 있는데 양의 차이가 아닌 등심과 상등심의 차이. 가격이 4천원이나 차이가 나지만 확실히 상이 더 쫄깃하면서 부드러움
👉밥 추가는 무료기 때문에 버터를 500원에 추가해서 자리마다 비치된 계란 간장을 밥에 뿌려 먹자!
"""
```

Create a document in which only nouns are extracted through a stemmer.
```python
okt = Okt()
tokenized_doc = okt.pos(doc)
tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])
print('품사 태깅 10개만 출력 :',tokenized_doc[:10])
print('명사 추출 :',tokenized_nouns)
```

```python
품사 태깅 10개만 출력 : [('[', 'Punctuation'), ('신사', 'Noun'), ('📍', 'Foreign'), ('카바', 'Noun'), ('동', 'Modifier'),
(']', 'Punctuation'),('\n', 'Foreign'), ('-', 'Punctuation'), ('\n', 'Foreign'), ('🍽', 'Foreign')]
명사 추출 : 신사 카바 레 카츠동 가라이 규동 컵누 신사역 신흥 강 카바 가로수길 건너편 위치 입 소문 중 내부 바 테이블 자리 좀 편이 회전 현재 웨이팅 처 메뉴인 레 카츠동 
돈까스 달달 레소스 계란 이불 밥 위 구성 돈까스 러버 라면 수 여기 꿀팁 미소 장국 닛신 컵누 변경 고기 라면 수 것 가라이 규동 매콤 규동 내 기름 추가 유독 볶음 맛 제육 다음 
방문 그냥 레 카츠동 주문 단무지 갓 유자 고추 테이블 갓 짭쪼름한 은근 별미 유자 고추 돈까스 찰떡 버터 추가 밥 것 왕왕 합리 가격 신사역 근처 밥 먹기 곳 더 핫 것 느낌 돈까스 자체 
아주 두툼 돈까스 과 달콤 레소스 계란 조화 레 카츠동 일반 상이 양 차이 등심 상등 심의 차이 가격 차이 나 상이 더 쫄깃 부드러움 밥 추가 무료 기 때문 버터 추가 자리 비치 계란 간장밥
```





Use sklearn CountVectorizer to extract words.
The reason for using CountVectorizer is, can easily extract n-grams by using the argument of n_gram_range. 
```python
n_gram_range = (2, 3)

count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
candidates = count.get_feature_names()

print('trigram 개수 :',len(candidates))
print('trigram 다섯개만 출력 :',candidates[:5])
```
```python
trigram 개수 : 203
trigram 다섯개만 출력 : ['가격 신사역', '가격 신사역 근처', '가격 차이', '가격 차이 상이', '가라이 규동']
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
    # 문서와 각 키워드들 간의 유사도
    distances = cosine_similarity(doc_embedding, candidate_embeddings)

    # 각 키워드들 간의 유사도
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # 각 키워드들 중에서 가장 덜 유사한 키워드들간의 조합을 계산
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
['유독 볶음 제육', '레소스 계란 이불', '테이블 짭쪼름한 은근', '달달 레소스 계란', '달콤 레소스 계란']
```
