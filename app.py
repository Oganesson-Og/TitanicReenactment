import numpy as np
from flask import Flask, render_template, request
import pickle


app = Flask(__name__)
model = pickle.load(open('titanic_model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    independent_set = [i for i in request.form.values()]
    independent_set = np.array(independent_set)
    independent_set = independent_set.reshape(1, -1)
    i = model.predict(independent_set)
    pred_dict = {'Sorry, this would have been your last day': 0, "Congrats, you are among the lucky survivors!": 1}
    pred_list = list(pred_dict)
    prediction = pred_list[int(i)]

    return render_template("index.html", prediction_text=prediction)


if __name__ == "__main__":
    app.run(debug=True)