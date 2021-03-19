import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)

#read the pickle file
model = pickle.load(open('model.pkl', 'rb'))
model_hdf = pickle.load(open('model_hdf.pkl', 'rb'))

@app.route('/')
def home():
	#generate the html file
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	#for rendering the results on HTML
	int_features = [x for x in request.form.values()]
	final_features = [np.array(int_features)]
	prediction = model.predict(final_features)

	output = math.exp(prediction[0])

	return render_template("index.html", prediction_text = 'The overstrain of the tool is {} minNm'.format(output))

@app.route('/predict_class', methods=['POST'])
def predict_class():
	int_features_hdf = [x for x in request.form.values()]
	final_features_hdf = [np.array(int_features_hdf)]
	prediction_class = model_hdf.predict(final_features_hdf)

	output_hdf = prediction_class[0]

	return render_template("index.html", prediction_hdf_text = 'The class is {}'.format(output_hdf))


if __name__=='__main__':
	app.run(debug=True)
