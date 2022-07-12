import pymysql
import pandas as pd
import struct
import numpy as np
import re
from ExtractReviewKeywords import ExtractMealionDBReviewKeywords


class DataPipelineForRecommendarSystem:

    def hangul_preprocessing(string):
        return re.compile('[^ ㄱ-ㅣ가-힣]+').sub(' ',  string)

    # mealionDB에서 reivew table select , 위도 경도값 부여
    def load_mealionDB_review():
        host = "  "  # 접속할 db의 host명
        user = "  "  # 접속할 db의 user명
        pw = "  "  # 접속할 db의 password
        conn = pymysql.connect(host=host,
                               user=user,
                               password=pw,
                               db="mealion")  # DB에 접속
        mealion_review_sql = "SELECT * FROM Reviews"
        cursor = conn.cursor()
        cursor.execute(mealion_review_sql)
        mealion_review = cursor.fetchall()
        mealion_review = pd.DataFrame(mealion_review)
        mealion_review.rename(columns={0: 'id', 1: 'content', 2: 'summary', 3: 'wouldVisitAgain', 4: 'authorId', 5: 'mentioningReiviewId',
                              6: 'restaurantId'	, 7: 'menus', 8: 'published', 9: 'feedback', 10: 'createdAt', 11: 'updatedAt'}, inplace=True)
        mealion_review = mealion_review[mealion_review['published'] == 1]
        # 만우절 이벤트 게시물 제거
        mealion_review = mealion_review[mealion_review['restaurantId'] != 850973]
        mealion_review.reset_index(drop=True, inplace=True)
        print('fisnish select reviewDB')
        mealion_restaurants_sql = "SELECT * FROM Restaurants;"
        cursor = conn.cursor()
        cursor.execute(mealion_restaurants_sql)
        mealion_restaurants = cursor.fetchall()
        mealion_restaurants = pd.DataFrame(mealion_restaurants)
        mealion_restaurants = mealion_restaurants.rename(columns={0: 'id', 1: 'name', 2: 'description', 3: 'breakTimeId', 4: 'address', 5: 'lotNumberAddress', 6: 'telephone',
                                                         7: 'location', 8: 'regionId', 9: 'ownerId', 10: 'profileImageColor', 11: 'password', 12: 'listed', 13: 'aredId', 14: 'createdAt', 15: 'updatedAt'})
        print('fisnish select restaurantDB')
        # 위경도 Byte to float 전환후 mealion_review dataframe에 add
        mealion_review['location'] = ''
        for i in range(len(mealion_review)):
            try:
                mealion_restaurants_index_number = mealion_restaurants[
                    mealion_restaurants['id'] == mealion_review['restaurantId'][i]].index[0]
                Byte = mealion_restaurants['location'][mealion_restaurants_index_number]
                New_lon_lat = struct.unpack('dd', Byte[-16:])
                mealion_review['location'][i] = New_lon_lat
            except:
                pass

        # 위경도 column 위도, 경도 각각 column divide
        mealion_review['lon'] = ''
        mealion_review['lat'] = ''
        for i in range(len(mealion_review)):
            try:
                before_lom_lat = str(mealion_review['location'][i])
                comma_number = before_lom_lat.find(',')
                mealion_review['lon'][i] = float(
                    before_lom_lat[1:comma_number])
                mealion_review['lat'][i] = float(
                    before_lom_lat[comma_number+1:-1])
            except:
                pass

        return mealion_review

    # review에 키워드값 부여
    def preparation_traing_data():
        mealion_review = DataPipelineForRecommendarSystem.load_mealionDB_review()
        mealion_review['recommmendar_traing_data'] = ''
        mealion_review['keyword'] = np.nan
        mealion_review['tag'] = np.nan
        # local file에서 read- review_keywords data
        local_mealion_data = pd.read_csv(
            'C:/Users/junseok/Downloads/local_mealion_data.csv')

        # keyword-> local에서 review로 전달
        for i in range(len(mealion_review)):
            try:
                local_mealion_data_index = local_mealion_data[mealion_review['id']
                                                              [i] == local_mealion_data['id']].index[0]  # Mealion 데이터 베이스에서 리뷰 제목, 식당 메뉴, 리뷰 태그를 가져왔다. 또한 Bert4key를 사용해 식당 리뷰에서 키워드를 추출했다.

                mealion_review['keyword'][i] = local_mealion_data['keyword_mmr_0.85_topn_5'][local_mealion_data_index]
                mealion_review['tag'][i] = local_mealion_data['tag'][local_mealion_data_index]
            except:
                pass

        # 키워드 없는 mealion review   indexing
        no_keywords_mealion_reviews = mealion_review[mealion_review['keyword'].isnull(
        )]
        # 키워드 추출 안되는 리뷰 제거
        # 키워드 추출해야 하는 리뷰 최신순으로 20개만 선택
        no_keywords_mealion_reviews = no_keywords_mealion_reviews[-20:]
        no_keywords_mealion_reviews.reset_index(drop=True, inplace=True)

        no_keywords_mealion_reviews, local_mealion_data = ExtractMealionDBReviewKeywords.extract_reviews_using_bert(
            no_keywords_mealion_reviews, local_mealion_data)  # local_mealion_data:키워드 및 새로운 review행 로컬파일로 저장 , no_keywords_mealion_reviews: 키워드 추출

        # no_keywords_mealion_reviews dataframe에서  mealion_review dataframe 으로 keyword 보내기
        for i in range(len(no_keywords_mealion_reviews)):
            reviewId = no_keywords_mealion_reviews['id'][i]
            mealion_review_index = mealion_review[mealion_review['id']
                                                  == reviewId].index[0]
            mealion_review['keyword'][mealion_review_index] = no_keywords_mealion_reviews['keyword'][i]

        # review에 태그값 부여

        # column들  null 값 제거
        mealion_review['keyword'].replace(np.nan, '', inplace=True)
        mealion_review['tag'].replace(np.nan, '', inplace=True)
        mealion_review['menus'].replace(np.nan, '', inplace=True)
        mealion_review['summary'].replace(np.nan, '', inplace=True)
        # bert 학습 데이터 column 생성
        for i in range(len(mealion_review)):
            mealion_review['recommmendar_traing_data'][i] = str(
                mealion_review['keyword'][i]) + ' ' + mealion_review['tag'][i] + ' ' + mealion_review['menus'][i] + ' ' + mealion_review['summary'][i]
        print('finsih set recommmendar_traing_data_column ')

        # recommmendar_traing_data column 한글만 남기기
        for i in range(len(mealion_review)):
            only_hangul = DataPipelineForRecommendarSystem.hangul_preprocessing(
                mealion_review['recommmendar_traing_data'][i])
            mealion_review['recommmendar_traing_data'][i] = only_hangul
        print('finish remain only hangul data')

        return mealion_review
