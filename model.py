import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pickle

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load your dataset (make sure it's in the correct format)
data = pd.read_csv('customer_reviews.csv')

# Handle missing or invalid data
# Fill NaN or missing reviews with a placeholder
data['Review'] = data['Review'].fillna("No review available")

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):  # Ensure the text is a string
        return ""  # Return empty string if not a valid text
    text = text.lower()  # Convert to lowercase
    stopwords = nltk.corpus.stopwords.words('english')  # Remove stopwords
    tokens = nltk.word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords]
    return ' '.join(tokens)

# Apply preprocessing to the reviews
data['cleaned_review'] = data['Review'].apply(preprocess_text)

# Split data into features (X) and labels (y)
X = data['cleaned_review']
y = data['Sentiment']  # Ensure this matches your sentiment column name

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a model pipeline: TF-IDF + Logistic Regression
model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))

# Train the model
model.fit(X_train, y_train)

# Save the trained model
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved as sentiment_model.pkl")
