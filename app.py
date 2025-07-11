from flask import Flask, request, render_template
from model import SentimentRecommenderModel

app = Flask(__name__)
sentiment_model = SentimentRecommenderModel()

@app.route('/')
def home():
    import os
    print("CWD:", os.getcwd())
    print("Templates contents:", os.listdir("templates") if os.path.exists("templates") else "Templates folder not found!")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    user = request.form['userName'].lower()
    items = sentiment_model.getSentimentRecommendations(user)

    if items is not None:
        return render_template("index.html", column_names=items.columns.values, row_data=list(items.values.tolist()), zip=zip, user=user)
    else:
        return render_template("index.html", message="User Name doesn't exist. No product recommendations at this point of time!")

@app.route('/predictSentiment', methods=['POST'])
def predict_sentiment():
    review_text = request.form["reviewText"]
    pred_sentiment = sentiment_model.classify_sentiment(review_text)
    return render_template("index.html", sentiment=pred_sentiment)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
