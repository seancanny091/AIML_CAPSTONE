## Using Text Classification to Determine a Person's Myers-Briggs Personality Type Indicator

### Executive Summary

**Project Overview and Goals:**
The project aims identify an effective way to determine an individual's Myers-Briggs Type Indicator (MBTI) based on their text posts, leveraging machine learning models to analyze and predict personality traits. This capability has applications in targeted marketing, personalized content, and psychological research.

**Findings:**
Through extensive data processing and model training, significant insights were gained into the relationship between text patterns and MBTI types. Focusing on the two most common MBTI types, INFP and INFJ, allowed for efficient model training and evaluation.

**Results and Conclusion:**
Several classification models were evaluated, including Logistic Regression, SVC, Decision Trees, Naive Bayes, and Random Forests. Both SentenceTransformer embeddings and TF-IDF vectorization approaches were used. The models showed promising results, with Random Forest and SVC providing the highest accuracy and balanced performance.

**Future Research and Development:**
Future work should focus on incorporating the full range of MBTI types and experimenting with advanced deep learning techniques to further improve prediction accuracy. Additionally, expanding the dataset and exploring other feature extraction methods could enhance model robustness.

**Next Steps and Recommendations:**
It is recommended to integrate the developed models into real-world applications for further validation and refinement. Continuous monitoring and updating of the models with new data will ensure sustained accuracy and relevance.

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
