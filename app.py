import os
import numpy as np
import sys
import caffe
from flask import Flask
from flask import jsonify
from flask import request
import requests
import json
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
		prediction = predict(image_path)
		resp = jsonify({"status": "cool", "prediction": prediction, "prediction_uuid": prediction_uuid, "image_s3_key": image_s3_key })
		resp.status_code = 200
		return resp
	except Exception as e:
		raise e


def predict(image_path, verbose=False):
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
	return cum_sum

#image_path = "demo_pic.png"
#for k in range(10):
#	print "Age : ", predict(image_path)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
