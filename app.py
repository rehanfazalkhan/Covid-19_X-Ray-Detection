
#::: Import modules and packages :::
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import numpy as np
from PIL import Image
import PIL
import os

#::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)

# ::: Prepare Keras Model :::
# Model files
#MODEL_ARCHITECTURE = './model/model_adam.json'
#MODEL_WEIGHTS = './model/model_100_eopchs_adam_20190807.h5'

# Load the model from external files
#json_file = open(MODEL_ARCHITECTURE)
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)
model=load_model("mot.h5")
#model._make_predict_function()
# Get weights into the model
#model.load_weights(MODEL_WEIGHTS)
#print('Model loaded. Check http://127.0.0.1:5000/')

# ::: MODEL FUNCTIONS :::
def model_predict(img_path, model):
	'''
		Args:
			-- img_path : an URL path where a given image is stored.
			-- model : a given Keras CNN model.
	'''

	IMG = image.load_img(img_path, target_size=(224, 224))
	# Pre-processing the image
	IMG_ = image.img_to_array(IMG)
	IMG_ = np.true_divide(IMG_, 255)
	IMG_ = np.expand_dims(IMG_,axis=0)
	model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
	prediction = model.predict(IMG_)
	return prediction
# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
	# Constants:
	classes = ["corona","no_corona"]
	classes=np.array(classes)

	if request.method == 'POST':

		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		# Make a prediction
		prediction = model_predict(file_path, model)
		i=np.argmax(prediction,axis=1)
		predicted_class=classes[i]
		#predicted_class = classes['TRAIN'][prediction[0]]
		#predicted_class = decode_predictions(preds, top=1)
		print('We think that is {}.'.format(predicted_class))

		return str(predicted_class)

if __name__ == '__main__':
	app.run(debug = True)