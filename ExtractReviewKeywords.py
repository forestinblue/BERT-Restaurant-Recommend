import numpy as np
import pandas as pd
import itertools
import time
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class ExtractMealionDBReviewKeywords:
    def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

        word_doc_similarity = cosine_similarity(
            candidate_embeddings, doc_embedding)  # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
        word_similarity = cosine_similarity(
            candidate_embeddings)  # 각 키워드들 간의 유사도
        # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
        keywords_idx = [np.argmax(word_doc_similarity)]
        # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

        # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복. ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
        for _ in range(top_n - 1):
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            target_similarities = np.max(
                word_similarity[candidates_idx][:, keywords_idx], axis=1)
            mmr = (1-diversity) * candidate_similarities - diversity * \
                target_similarities.reshape(-1, 1)  # MMR을 계산
            mmr_idx = candidates_idx[np.argmax(mmr)]
            keywords_idx.append(mmr_idx)  # keywords & candidates를 업데이트
            candidates_idx.remove(mmr_idx)

        return [words[idx] for idx in keywords_idx]

    def max_sum_sim(doc_embedding, candidate_embeddings, words, top_n, nr_candidates):

        distances = cosine_similarity(
            doc_embedding, candidate_embeddings)  # 문서와 각 키워드들 간의 유사도
        distances_candidates = cosine_similarity(candidate_embeddings,  # 각 키워드들 간의 유사도
                                                 candidate_embeddings)
        # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
        words_idx = list(distances.argsort()[0][-nr_candidates:])
        words_vals = [candidates[index] for index in words_idx]
        distances_candidates = distances_candidates[np.ix_(
            words_idx, words_idx)]
        min_sim = np.inf
        candidate = None

        # 각 키워드들 중에서 가장 덜 유사한 키워드들간의 조합을 계산
        for combination in itertools.combinations(range(len(words_idx)), top_n):
            sim = sum([distances_candidates[i][j]
                      for i in combination for j in combination if i != j])
            if sim < min_sim:
                candidate = combination
                min_sim = sim

        return [words_vals[idx] for idx in candidate]

    def extract_reviews_using_bert(no_keywords_mealion_reviews, local_mealion_data):
        print('start extract')
        print("len(no_keywords_mealion_reviews) : {}".format(
            len(no_keywords_mealion_reviews)))
        newdata_to_localfile_count = 0
        olddata_to_localfile_count = 0
        for i in range(len(no_keywords_mealion_reviews)):

            print(i)

            keyword_mmr_7_topn_5 = ''
            keyword_mmr_85_topn_5 = ''
            keyword_max_sum_sim_ngram_13_topn_5 = ''
            keyword_max_sum_sim_ngram_13_topn_3 = ''

            try:

                doc = no_keywords_mealion_reviews['content'][i]

                okt = Okt()
                tokenized_doc = okt.pos(doc)
                tokenized_nouns = ' '.join(
                    [word[0] for word in tokenized_doc if word[1] == 'Noun'])

                n_gram_range = (1, 3)
                count = CountVectorizer(
                    ngram_range=n_gram_range).fit([tokenized_nouns])
                global candidates
                candidates = count.get_feature_names()

                time.sleep(1)
                model = SentenceTransformer(
                    'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
                doc_embedding = model.encode([doc])
                candidate_embeddings = model.encode(candidates)
                time.sleep(1)
                distances = cosine_similarity(
                    doc_embedding, candidate_embeddings)
                try:
                    keyword_mmr_7_topn_5 = ExtractMealionDBReviewKeywords.mmr(
                        doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.7)
                except:
                    keyword_mmr_7_topn_5 = ''
                    pass
                try:
                    keyword_mmr_85_topn_5 = ExtractMealionDBReviewKeywords.mmr(
                        doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.85)
                except:
                    keyword_mmr_85_topn_5 = ''
                    pass
                try:
                    keyword_max_sum_sim_ngram_13_topn_5 = ExtractMealionDBReviewKeywords.max_sum_sim(
                        doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=30)
                except:
                    keyword_max_sum_sim_ngram_13_topn_5 = ''
                    pass
                try:

                    keyword_max_sum_sim_ngram_13_topn_3 = ExtractMealionDBReviewKeywords.max_sum_sim(
                        doc_embedding, candidate_embeddings, candidates, top_n=3, nr_candidates=30)
                except:
                    keyword_max_sum_sim_ngram_13_topn_3
                    pass

                no_keywords_mealion_reviews['keyword'][i] = keyword_mmr_85_topn_5

            except:
                continue

            try:

                local_mealion_index = local_mealion_data[local_mealion_data['id'] == i].index[0]
                # local_mealion_index 존재하면 , local mealion data에 해당 review id가 존재한다는 뜻
                # local mealion data의 해당 review id에 키워드를 넣기
                local_mealion_data['keyword_max_sum_sim_ngram_13_topn_3'][local_mealion_index] = keyword_max_sum_sim_ngram_13_topn_3
                local_mealion_data['keyword_max_sum_sim_ngram_13_topn_5'][local_mealion_index] = keyword_max_sum_sim_ngram_13_topn_5
                local_mealion_data['keyword_mmr_7_topn_5'][local_mealion_index] = keyword_mmr_7_topn_5
                local_mealion_data['keyword_mmr_85_topn_5'][local_mealion_index] = keyword_mmr_85_topn_5
                newdata_to_localfile_count += 1

            except:

                # local_mealion_index 존재하지 않으면, local mealion data에 해당 review id가 존재하지 않는다.
                # 새로운 reviewid라는 뜻
                # local mealion data에  새로운 행 추가 , 키워드 모두 저장
                mealion_review_new_row_data = {'id':  no_keywords_mealion_reviews['id'][i],
                                               'content':  no_keywords_mealion_reviews['content'][i],
                                               'summary':  no_keywords_mealion_reviews['summary'][i],
                                               'wouldVisitAgain':  no_keywords_mealion_reviews['wouldVisitAgain'][i],
                                               'authorId':  no_keywords_mealion_reviews['authorId'][i],
                                               'mentioningReiviewId':  no_keywords_mealion_reviews['mentioningReiviewId'][i],
                                               'restaurantId': no_keywords_mealion_reviews['restaurantId'][i]	,
                                               'menus':  no_keywords_mealion_reviews['menus'][i],
                                               'published':  no_keywords_mealion_reviews['published'][i],
                                               'feedback':  no_keywords_mealion_reviews['feedback'][i],
                                               'createdAt':  no_keywords_mealion_reviews['createdAt'][i],
                                               'updatedAt': no_keywords_mealion_reviews['updatedAt'][i],
                                               'keyword_max_sum_sim_ngram_13_topn_3': keyword_max_sum_sim_ngram_13_topn_3,
                                               'keyword_max_sum_sim_ngram_13_topn_5': keyword_max_sum_sim_ngram_13_topn_5,
                                               'keyword_mmr_7_topn_5': keyword_mmr_7_topn_5,
                                               'keyword_mmr_85_topn_5': keyword_mmr_85_topn_5}
                local_mealion_data.append(
                    mealion_review_new_row_data, ignore_index=True)
                olddata_to_localfile_count += 1
        print('save_olddata_to_localfile_count: {}'.format(
            olddata_to_localfile_count))
        print('save_newdata_to_localfile_count: {}'.format(
            newdata_to_localfile_count))
        print('finish extract')

        local_mealion_data.reset_index(drop=True, inplace=True)
        # 새로운 키워드 , 리뷰 데이터 받은 local_mealion_data 저장
        local_mealion_data.to_csv(
            'C:/Users/junseok/Downloads/local_mealion_data.csv', index=False)
        return no_keywords_mealion_reviews, local_mealion_data
