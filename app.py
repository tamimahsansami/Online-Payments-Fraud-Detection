import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np  # Added import

app = Flask(__name__)

# Corrected variable names
decision_model = pickle.load(open('decision_model.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    type_mapping = {
        "PAYMENT": 2,
        "TRANSFER": 4,
        "CASH_OUT": 1,
        "DEBIT": 5,
        "CASH_IN": 3
    }
    data['type'] = type_mapping.get(data['type'], -1)
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = decision_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)