from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# Expanded sample dataset
data = [
    ("I love this product!", "positive"),
    ("Absolutely fantastic, works perfectly.", "positive"),
    ("Very bad experience.", "negative"),
    ("Terrible quality and waste of money.", "negative"),
    ("Poor quality", "negative"),
    ("The product is okay.", "neutral"),
    ("It’s fine but not what I expected.", "neutral"),
    ("Does the job, nothing special.", "neutral"),
    ("Amazing quality and performance!", "positive"),
    ("Worst thing I ever bought.", "negative"),
    ("Not great, not terrible.", "neutral"),
    ("Nice product but could be better.", "neutral"),
    ("Product is excellent and works like a charm.", "positive"),
    ("Quality is very low and disappointing.", "negative"),
    ("I wouldn’t buy it again, but it works.", "neutral"),
]

texts, labels = zip(*data)

# Model pipeline with TfidfVectorizer
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(texts, labels)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    prediction = model.predict([review])[0]
    return render_template('index.html', review=review, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
