## Using Text Classification to Determine a Person's Myers-Briggs Personality Type Indicator

### Executive Summary

**Project Overview and Goals:**
The goal of this project is to identify an effective way to determine a person's Myers-Briggs Type Indicator (MBTI) based on their text posts. We will be training and tuning several classification models to accurately classify social media posts according to MBTI types. Models will be able to predict, from future/unseen posts, the MBTI type of their writer. We will then evaluate and compare the models' performances to identify the best one and further scrutinize it to find the most effective features (words) that enhance performance in this classification task. Insights will be drawn from this model by conducting a global analysis, using various libraries to identify the most important words/features used for making accurate predictions. We will also be locally analyzing this model and evaluating its class prediction process for individual posts. Lastly, we will draw insights from our analyses and recommend areas to research and courses to undertake for future work in determining MBTI from text posts.

**Findings:**
The best model for determining a person's Myers-Briggs Type Indicator (MBTI) from text posts is the Support Vector Classifier (SVC) model, using Lemmatization + TF-IDF, with an accuracy score of 0.5735, a recall of 0.5735, and an F1 score of 0.572964. Its performance is followed by the Random Forest model, Logistic Regression model, and the Naive Bayes model. This decision is based on comparing the finetuned models' accuracy, recall, and F1 scores (results summary below). The SVC model has the best overall performance metrics, including accuracy, recall, and F1 score. In terms of errors, the Logistic Regression model has a balanced number of false positives (FP) and false negatives (FN), the Naive Bayes model has more FPs than FNs, the Decision Tree model has a similar count of FP and FN, and the SVC model maintains a slightly higher number of FPs compared to FNs.

**Results and Conclusion:**
In this project, we explored two different approaches to determine a person's Myers-Briggs Type Indicator (MBTI) from text posts: SentenceTransformer embeddings and Lemmatization + TF-IDF vectorization.

The Lemmatization + TF-IDF approach outperformed the SentenceTransformer approach across all models. The Support Vector Classifier (SVC) using Lemmatization + TF-IDF achieved the highest accuracy score of 0.5735, recall of 0.5735, and F1 score of 0.572964. This was followed by the Decision Tree and Random Forest models, which also showed strong performance with accuracy scores of 0.5710 and 0.5665, respectively.

In contrast, the SentenceTransformer approach did not perform as well. The highest accuracy was observed with the SVC model, achieving a score of 0.5525, while the Random Forest model achieved an accuracy of 0.5515. These results indicate that the Lemmatization + TF-IDF approach captures the textual features more effectively for this particular classification task.

While the Lemmatization + TF-IDF approach provides a more effective feature representation for MBTI classification from text posts compared to SentenceTransformer, neither approach achieved sufficiently high accuracy to be used reliably in real-life applications. The highest accuracy of 0.5735 indicates that there is still significant room for improvement in predicting MBTI types from text.

**Next Steps and Recommendations:**
Leverage Deep Learning Models:

* Neural Networks: CNNs or RNNs could capture more complex text patterns.
* Transformers: Models like BERT or GPT, known for state-of-the-art NLP performance, could enhance results.

Hybrid Approaches:
* Combine TF-IDF with neural embeddings for richer text representation.

Data Augmentation:
* Increase dataset size and diversity with techniques like paraphrasing and back-translation.

Ensemble Methods:
* Use ensemble techniques like stacking or boosting to improve robustness and accuracy.

Feature Engineering:
* Explore additional features like sentiment analysis, topic modeling, and linguistic features.

By implementing these advanced techniques, we can aim for more accurate and reliable MBTI prediction models suitable for real-life applications.

### Rationale
Understanding a person's Myers-Briggs Type Indicator (MBTI) from their text posts provides numerous compelling benefits and applications. Utilizing this capability allows organizations and researchers to gain deeper insights into individual personality traits, which can be applied in various domains. This opens up a wide range of opportunities across multiple fields. From personalized marketing and enhanced user experiences to improved team dynamics and advanced psychological research, the potential applications are extensive and impactful. By recognizing and addressing the diverse personality traits of individuals, organizations can achieve higher engagement, satisfaction, and success.

### Data Sources
**Dataset**
The dataset was sourced from Kaggle, containing over 8600 rows of MBTI types and their corresponding social media posts. Each entry provided a rich text-based profile for analysis.

**Exploratory Data Analysis:**
Initial exploration involved understanding the data distribution, characterizing the length of posts, and identifying the unique MBTI types. Visualizations highlighted the imbalance among different personality types, guiding subsequent preprocessing steps.

**Cleaning and Preparation:**
Data cleaning involved converting text to lowercase, parsing posts into separate rows, and removing URLs, special characters, and numbers. This ensured that the textual data was uniform and ready for analysis.

**Preprocessing:**
The dataset was filtered to focus on INFP and INFJ types, balancing the data to improve model performance. Text data was then prepared using two methods: SentenceTransformer for generating embeddings and TF-IDF vectorization for capturing term frequency and importance.

**Final Dataset:**
The final dataset was balanced and preprocessed, containing clean text posts with associated MBTI types. This prepared data was then used for model training and evaluation.

### Methodology

**SentenceTransformer:**
The SentenceTransformer model was used to generate semantic embeddings from the text posts, providing a context-aware representation of the data.

**Lemmatization and TFIDF:**
Text posts were lemmatized to their base forms, and TF-IDF vectorization was applied to capture the importance of terms within the posts.

**Logistic Regression:**
This model was tuned using GridSearchCV, evaluating different penalties and solvers to optimize performance on the training set.

**SVC:**
Support Vector Classifier was applied with linear and RBF kernels, aiming to find the best separating hyperplane for the MBTI types.

**Decision Tree:**
Decision Tree models were evaluated with varying depths and minimum samples splits to determine the best structure for classification.

**Naive Bayes:**
Both Gaussian and Multinomial Naive Bayes models were tested, focusing on different assumptions about the distribution of features.

**Random Forest:**
Random Forest models were trained with various numbers of estimators and depths, leveraging ensemble learning to enhance predictive accuracy.

### Model Evaluation and Results

The models were evaluated based on accuracy, recall, F1 score, and confusion matrices. Results showed that the Random Forest and SVC models provided the highest accuracy and balanced performance. Confusion matrices for each model illustrated their ability to correctly classify INFP and INFJ types, with detailed classification reports highlighting the precision and recall for each class. The use of both SentenceTransformer embeddings and TF-IDF vectorization provided a comprehensive evaluation, ensuring that the models captured both semantic meaning and term importance.
