# Sentiment Based Product Recommendation

## Problem Statement

The landscape of modern commerce has undergone a profound transformation, with e-commerce becoming a dominant force. Traditional brick-and-mortar models have given way to online platforms, enabling companies to connect directly with consumers. Industry leaders like Amazon and Flipkart have set the benchmark by offering extensive product selections at users’ fingertips.

In this rapidly growing space, companies like Ebuss are tapping into the momentum, building a strong presence across diverse product categories—from household items to electronics—to capture a significant share of the market.

However, thriving in such a competitive environment requires continuous innovation. To stand out, Ebuss must not only keep up with market leaders but strive to lead through technology-driven solutions that elevate the user experience.

As a Machine Learning Engineer at Ebuss, your mission is to develop a robust, sentiment-driven recommendation system that enhances product suggestions based on user feedback. This initiative involves several critical components:

 

#### Step 1: Exploratory Data Analysis (EDA) & Preprocessing


1) Understand the Dataset: Analyze shape, null values, unique counts.

2) Clean the Dataset: Remove duplicates, handle missing values, and standardize text fields.

3) Visualizations: Plot rating distribution, sentiment distribution, top users/products, etc.

4) Text Preprocessing: Apply lowercasing, remove punctuation and stopwords, perform lemmatization.
 


#### Step 2: Sentiment Analysis

1) Vectorization: Use TF-IDF or Bag-of-Words on reviews_text.

2) Model Training: Train any 3 of the following models:

3) Logistic Regression

4) Naive Bayes

5) Random Forest

6) XGBoost

7) Hyperparameter Tuning: Optimize model performance.

8) Evaluation: Use accuracy, precision, recall, and F1-score.

9) Best Model Selection: Choose the top-performing model for further use.


#### Step 3: Recommendation System

1) Collaborative Filtering: Implement User-User or Item-Item filtering using cosine similarity.

2) Top 20 Product Recommendations: Generate 20 product suggestions per user based on ratings.

3) Sentiment-Boosted Filtering: Refine recommendations by selecting the top 5 products with the most positive sentiment from the initial 20.


### Step 4: Deployment

1) Flask Web App:

	Input: Username

	Output: 5 personalized product recommendations

2) Deployment on Railway: Integrate the UI, sentiment model, and recommendation logic into a live application