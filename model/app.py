from flask import Flask, render_template, request, jsonify
#ช่วยให้รีซอร์สสารรมารถทำงานบนโดเมนอื่นที่เป็นส่งนขยายได้
from flask_cors import CORS
from chat import get_response

app = Flask(__name__)
CORS(app)


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    #TODO: check if text is valid
    response = get_response(text)
    massage = {"answer": response}
    return jsonify(massage)

if __name__ =="__main__":
    app.run(debug=True)