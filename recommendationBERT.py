from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from haversine import haversine
from datapipeline_mealionDB import DataPipelineForRecommendarSystem


class RecommendationBERT:
    def training_bert():
        global mealion_review
        mealion_review = DataPipelineForRecommendarSystem.preparation_traing_data()
        print('finish preparation_traing_data')
        bert = SentenceTransformer('bert-base-nli-mean-tokens')
        sentence_embeddings = bert.encode(
            mealion_review['recommmendar_traing_data'].tolist())
        print('fisish bert_sentence_embeddings')
        global similarity  # 전역 변수 지정
        similarity = cosine_similarity(sentence_embeddings)

        return mealion_review, similarity

    def recommendar_bert(request_indexId, request_authorId):
        index = request_indexId
        restaurant_recommendation = sorted(
            list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)
        Pin_restaurant_lon = mealion_review.iloc[index]['lon']
        Pin_restaurant_lat = mealion_review.iloc[index]['lat']
        # Latitude, Longitude
        Pin_restaurant = (Pin_restaurant_lat, Pin_restaurant_lon)
        print(Pin_restaurant)
        recommander_restaurant_dataframe = pd.DataFrame(
            {'review_id': [], 'review_similarity': []})
        # 거리 계산

        restaurantID = mealion_review['restaurantId'].iloc[index]
        # 같은 restaurantID인 review dataframe
        same_restaurantId_dataframe = mealion_review[mealion_review['restaurantId'] == restaurantID]
        delect_reviewID_list_same_restaurantId = same_restaurantId_dataframe['id'].to_list(
        )
        # 같은 authorID인 review dataframe
        same_userId_dataframe = mealion_review[mealion_review['authorId']
                                               == request_authorId]
        delect_reviewID_list_same_userID = same_userId_dataframe['id'].to_list(
        )
        delect_reviewID_list = delect_reviewID_list_same_userID + \
            delect_reviewID_list_same_restaurantId  # 추천할 수 없는  reviewID List

        dataframe_append_count = 0
        # 자기 자신 피드 id는 제외, 자기 자신 피드 similarity값: 1  가장 높음
        for i in range(1, len(mealion_review)):
            review_similarity = restaurant_recommendation[i][1]
            if review_similarity > 0.9835:  # 리뷰 유사도 0.9835 점 이상만 추출

                recommander_restaurant_lon = mealion_review.iloc[restaurant_recommendation[i][0]]['lon']
                recommander_restaurant_lat = mealion_review.iloc[restaurant_recommendation[i][0]]['lat']
                recommander_restaurant = (
                    recommander_restaurant_lat, recommander_restaurant_lon)
                review_restaurant_distance = haversine(
                    Pin_restaurant, recommander_restaurant, unit='km')

                if review_restaurant_distance < 5:  # 리뷰 식당과 5km이내인 식당만 추출

                    review_index = restaurant_recommendation[i][0]
                    review_id = int(mealion_review.iloc[review_index]['id'])
                    if review_id in delect_reviewID_list:  # reviewID가 추천 할 수 없는  reviewID이면 pass
                        continue
                    review_similarity = restaurant_recommendation[i][1]
                    recommander_restaurant_dataframe = recommander_restaurant_dataframe.append(
                        {'review_id': review_id, 'review_similarity': review_similarity}, ignore_index=True)
                    dataframe_append_count += 1
                    if dataframe_append_count >= 6:  # dataframe row가 5개 이상이면  break
                        break

            recommander_restaurant_list = recommander_restaurant_dataframe['review_id'].to_list(
            )
        return recommander_restaurant_list
