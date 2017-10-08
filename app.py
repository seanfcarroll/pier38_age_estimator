import os
import numpy as np
import sys
import caffe
from flask import Flask
from flask import jsonify
from flask import request
import requests
import json

import boto3
import io
import tempfile
import json
import os
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)

#caffe.set_mode_gpu()
caffe.set_mode_cpu()

mean_filename=os.path.join('imagenet_mean.binaryproto')
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0].mean(1).mean(1)

net_pretrained = os.path.join("dex_imdb_wiki.caffemodel")
net_model_file = os.path.join("age.prototxt")
Net = caffe.Classifier(net_model_file, net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

audio_model = LinearRegression()
### hard code linear coefficiencts ###
audio_model.coef_ = np.array([-0.00414669])
audio_model.intercept_ = 98.2135216968

@app.route('/',methods=['GET'])
def index():
    return "<h1>Advantage Age Prediction<h1><p>Hello, world!</p>"

@app.route('/api/predict',methods=['GET'])
def api_predict():
	try:
		payload = request.json
		prediction_uuid = request.json.get('prediction_uuid')
		image_s3_key = request.json.get('image_s3_key')

		image_path = "demo_pic.png"
		prediction = predict(image_path_s3)
		resp = jsonify({"status": "cool", "prediction": prediction, "prediction_uuid": prediction_uuid, "image_s3_key": image_s3_key })
		resp.status_code = 200
		return resp
	except Exception as e:
		raise e


@app.route('/api/audio',methods=['GET'])
def api_audio():
	try:
		audio_threshold = request.json.get('audio_threshold')
		x = [[audio_threshold]]
		prediction = audio_model.predict(x)
		prediction = 25
		resp = jsonify({"prediction": prediction})
		resp.status_code = 200
		return resp
	except Exception as e:
		raise e


def predict(image_path_s3, verbose=False):
    	image_path = s3_to_tempfile(image_path_s3)
	input_image = caffe.io.load_image(image_path)
	prediction = Net.predict([input_image],oversample=False)

	if verbose: 
		print "="*100
	cum_sum = 0
	for _idx, val in enumerate(prediction[0]):
		if verbose: print _idx , ": ", val*100,"%"
		cum_sum += _idx*val
	if verbose:
		print "="*100
		print 'predicted category is {0}'.format(prediction.argmax())
		print "Weighted mean prediction ", cum_sum
		print "Integreified Weighted mean prediction ", int(cum_sum)
	return prediction.argmax()

def s3_to_tempfile(key, _dtype):
  s3_obj = get_s3_obj(key)
  f = tempfile.NamedTemporaryFile(delete=False)
  f.write(s3_obj.read()) 
  f.seek(0)
  return f.name

def get_s3_obj(key):
  try:
    s3 = boto3.client('s3',aws_access_key_id=config.AWS_ACCESS_KEY_ID,aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY)
    #s3_url = '{}/{}/{}'.format(s3.meta.endpoint_url, BUCKET, key)
    s3_obj = s3.get_object(Bucket=config.BUCKET, Key=key)['Body']
    app.logger.info("Downloading S3 key: %s " % key)
    return s3_obj
  except Exception as e:
    app.logger.info(e)
    raise e

#image_path = "demo_pic.png"
#for k in range(10):
#	print "Age : ", predict(image_path)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
