# Customer_Review_Analysis
This project is a sentiment analysis web application that classifies customer reviews as positive, negative, or neutral using machine learning.

Here’s a simple README for your project:

---

# Customer Review Sentiment Analysis 🌟

This project performs sentiment analysis on customer reviews to predict whether the reviews are **positive**, **negative**, or **neutral**. It uses a machine learning model to classify the sentiments based on the input review text. The model is trained using the **Flipkart Product Customer Reviews** dataset from Kaggle.

## 🚀 Features

- Classifies reviews into three categories: **Positive**, **Negative**, and **Neutral**.
- Built with **Flask** for easy web deployment.
- Includes a simple web interface to input product reviews and see predictions.

## 🎯 Objective

To help businesses analyze customer feedback by automating the sentiment analysis process. This tool can classify product reviews based on their sentiment (positive, negative, or neutral).

## 📂 Dataset

The dataset used in this project is from Kaggle:

**[Flipkart Product Customer Reviews Dataset](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset)**

It contains customer reviews for various Flipkart products, and the sentiment of each review is labeled. We use this dataset to train the model.

## 💻 Technologies Used

- **Python** 🐍
- **Flask** for web deployment 🖥️
- **NLTK** for natural language processing 🧠
- **Scikit-learn** for machine learning 🔧
- **HTML/CSS** for frontend development 🌐

## 🔧 Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Gobika-R/Customer_Review_Analysis.git
   cd Customer_Review_Analysis
   ```

2. **Create a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Run the following command to install all necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset**:
   Download the **Flipkart Product Customer Reviews** dataset from Kaggle:
   [Download Dataset](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset)
   - Place the dataset in the `data/` directory of the project.

5. **Train the Model**:
   Before running the application, train the sentiment analysis model:
   ```bash
   python train_model.py
   ```
   This will generate the model file `sentiment_model.pkl`.

6. **Run the Application**:
   To run the Flask application locally:
   ```bash
   python app.py
   ```
   The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## 📸 Web Interface

Once the app is running, you can visit the webpage where you can:
1. Enter a review in the text input field.
2. Click on **Submit** to predict whether the review is positive, negative, or neutral.
3. View the sentiment result displayed on the page.

## 🌟 Example Reviews & Predictions

- **Positive Review**: "Great product! Worth the price."
  - **Prediction**: Positive 😄

- **Negative Review**: "The quality is very poor. Not recommended!"
  - **Prediction**: Negative 😞

- **Neutral Review**: "The product works fine, but not as expected."
  - **Prediction**: Neutral 😐

## 🛠️ How the Model Works

1. **Data Preprocessing**: The dataset is cleaned by removing stopwords, punctuation, and performing tokenization.
2. **Feature Extraction**: The text data is converted into numerical features using **TF-IDF** (Term Frequency-Inverse Document Frequency).
3. **Model Training**: A **Naive Bayes classifier** is trained using the processed data.
4. **Sentiment Prediction**: The trained model is used to predict the sentiment of any input review.

## 🚀 Deployment

To deploy this app on a cloud platform (like Heroku or AWS), follow these steps:

1. Create a **requirements.txt** file if not present:
   ```bash
   pip freeze > requirements.txt
   ```

2. Push the code to your GitHub repository and link it to a cloud platform (e.g., Heroku).

3. Set up your cloud platform with Python runtime and dependencies.

4. Push your code to the cloud platform and start the application.

## 📝 Contributions

Feel free to fork this project and contribute! Pull requests are always welcome. 😄

---

Let me know if you need any changes or additions to this README! 😊
