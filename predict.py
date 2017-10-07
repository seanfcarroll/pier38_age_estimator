import os
import numpy as np

import sys
"""
If you do not have caffe root setup

caffe_root = '~/caffe/' #Path to you caffe root 
sys.path.insert(0, caffe_root + 'python')
"""

import caffe
# NOTE: Comment the line below, if you want to run in CPU mode
#caffe.set_mode_gpu()
caffe.set_mode_cpu()

# age.prototxt			dex_imdb_wiki.caffemodel	imagenet_mean.binaryproto


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


def predict(image_path, verbose=False):
	input_image = caffe.io.load_image(image_path)
	prediction = Net.predict([input_image],oversample=False)

	if verbose: print "="*100
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

image_path = "demo_pic.png"
for k in range(10):
	print "Age : ", predict(image_path)
