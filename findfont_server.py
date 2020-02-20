from flask import Flask, jsonify, Response
import json
from functools import wraps
from flask_restful import Resource, Api, reqparse
# 이미지 처리를 위한 모듈 임포트
import werkzeug # multipart 형태의 데이터 타입을 지정하기 위해( FileStorage )
import numpy as np
import cv2
# 텐서플로우 사용
import tensorflow as tf
import findfont.image_handler as ff

app = Flask(__name__)
api = Api(app)


def as_json(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        res = f(*args, **kwargs)
        res = json.dumps(res, ensure_ascii=False).encode('utf8')
        return Response(res, content_type='application/json; charset=utf-8')
    return decorated_function


# 예측 함수
def predict_by_keras_model(_imageFileStream):
    # 1) 이미지 배열 만들기
    predict_image_array = ff.image_for_predict(_imageFileStream)
    # 2) 모델 불러오기
    model = tf.keras.models.load_model('256x3-CNN.model')
    # 3) 모델과 데이터를 이용해 예측
    CATEGORIES = ['Dog', 'Cat']
    prediction = model.predict([predict_image_array])
    print('hello')
    return CATEGORIES[int(prediction[0][0])]
# Json 형태의 데이터를 받기 위해서는
# post 방식의 요청 처리가 필요하다.


class ReceiveJson1(Resource):
    def post(self):
        try:
            # 파라미터를 받기 위한 작업
            parser = reqparse.RequestParser()
            # 받아낼 파라미터의 키값 및 타입 정의
            parser.add_argument('email', type=str)
            parser.add_argument('user_name', type=str)
            parser.add_argument('password', type=str)
            args = parser.parse_args() # 실제 데이터를 받아내는 구간
            # 파라미터에서 데이터를 각각 꺼내기
            _userEmail    = args['email']
            _userName     = args['user_name']
            _userPassword = args['password']
            # 사용자가 보내는 파라미터를 제일 먼저 수신 해야 하고
            # 그 다음 필요한 작업(예측, 집계, 추천 등등...)을 한다.
            return { 'Email' : _userEmail,
                      'UserName' : _userName,
                      'Password' : _userPassword,
                      'status' : 'success'}
        except Exception as e:
            return {'error' : str(e)}
# 이미지는 Json이 아님.. => binary 형태로 데이터가 들어옴 => stream을 통과함
# 무조건 이미지는 post..
# 이미지 또는 기타 파일들은 multipart-form 이라고 명명한다.


class ReceiveImageData(Resource):
    @as_json
    def post(self):
        try:
            print('hi')
            parser = reqparse.RequestParser()
            # 첫 번째는 키값,
            parser.add_argument('picture',
                                type=werkzeug.datastructures.FileStorage,
                                location='files')
            args = parser.parse_args()
            _imgFileStream = args['picture'].stream

            #cv2.imread랑 같음.
            _encodedArrayImage = ff.load_image(_imgFileStream)
            #############################
            # 매칭 알고리즘 수행 #
            image1, detected_text = ff.capture_image(_encodedArrayImage.copy())
            print("converted image : {}".format(image1))
            # sort_value = ff.target_image(image1, detected_text)
            sort_value = ff.target_image(image1, "가")
            ##############################

            # 매칭 알고리즘 결과 return

            return {"result": sort_value}
        except Exception as e:
            print("error : ", e)
            return {"error": str(e)}


api.add_resource(ReceiveJson1, '/rcv_sample1')
api.add_resource(ReceiveImageData, '/predict_font')
if __name__ == '__main__':
    print("===================Server Start===================")
    app.run(debug=True)