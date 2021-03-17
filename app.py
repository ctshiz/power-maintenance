import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

#read the pickle file
model = pickle.load(open('model.pkl', 'rb'))

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

	output = prediction[0]

	return render_template("index.html", prediction_text = 'The machine power is {} W'.format(output))

if __name__=='__main__':
	app.run(debug=True)
