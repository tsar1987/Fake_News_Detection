# Binary Classification of Fake News and Real News
## Introduction
This project aims to develop a robust binary classification model to distinguish between fake news and real news articles.      
Using traditional machine learning method and NLP method and compare these two methods
## Audience
Journalists, editors, and media analysts who are interested in understanding and combating the spread of fake news.      
General Public: People interested in understanding how fake news is detected and differentiated from real news, as well as those concerned about the impact of misinformation on society.      
Researchers, scholars, and students interested in machine learning, natural language processing, or media studies.     
## Data
Train Dataset: 2 csv files from Kaggle      
fake news:   
23481 rows and 5 columns, including information about title, text, subject, date and label.
real news:   
21417 rows and 5 columns, including information about title, text, subject, date and label.
https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection      
Real_world unseen test set, 1 csv files      
Collect 25 pieces of news from internet.      
the 13 real news comes from reuters (7 news), cnn (2 news) and npr(4 news),      
the 12 real news comes from breitbart (5 news) and thegatewaypundit (7 news).   
## Exploratory Data Analysis

![image](https://github.com/tsar1987/Fake_News_Detection/blob/94835aae49ae36d10e763f4cb7774452c13d4127/Figure/news%20proportion.png)

Fake news 54.5%     Real news 45.5%    

![image](https://github.com/tsar1987/Fake_News_Detection/blob/94835aae49ae36d10e763f4cb7774452c13d4127/Figure/year.png)

News distribution by year is biased. 2015 and 2018 only fake news.        

![image](https://github.com/tsar1987/Fake_News_Detection/blob/94835aae49ae36d10e763f4cb7774452c13d4127/Figure/month.png)

Patterns of news distribution by months: more real news from Sept. to Dec.

![image](https://github.com/tsar1987/Fake_News_Detection/blob/94835aae49ae36d10e763f4cb7774452c13d4127/Figure/day_of_week.png)

Patterns of news distribution by day: more fake news at weekends

![image](https://github.com/tsar1987/Fake_News_Detection/blob/94835aae49ae36d10e763f4cb7774452c13d4127/Figure/day_to_election.png)

Distribution of news by day to election: more fake news before election

## Data Training and Modeling
Hyperparameter Table:

![image](https://github.com/user-attachments/assets/23131aa6-008f-41b7-b35b-6d248ca8c7da)

Based on the comparative performance of the four initial models, the **XGBoost classifier** has been selected for further tuning and optimization. In this first round of experiments, which combined text and metadata features, the XGBoost model consistently outperformed Logistic Regression, Random Forest, and Gradient Boosting. It achieved the highest scores across all key evaluation metrics on the validation set, including a top validation accuracy of 0.987, a recall of 0.983, a precision of 0.989, and a leading F1-score of 0.986. This superior and well-balanced performance makes it the most robust and promising candidate for subsequent hyperparameter tuning. 

Best Model Performance on Test Set       
![image](https://github.com/user-attachments/assets/8673bc50-efb7-4c34-ba86-e5bc166aa506)

The results are quite revealing, as they demonstrate that significant tuning of the CountVectorizer parameters yielded no improvement whatsoever over the baseline model. Despite implementing more sophisticated text feature engineering—by including bi-grams and tri-grams, limiting the vocabulary to the top 2000 features, and filtering out both very common and very rare words—the model's performance on the test set remained absolutely identical across all metrics: accuracy, recall, precision, and F1-score. This compelling result strongly suggests that the model's predictive power is overwhelmingly derived from the non-text metadata features. The XGBoost algorithm is likely relying so heavily on signals from features like character counts, word counts, and days to election that changes in the text representation have become effectively negligible, indicating a point of diminishing returns for text feature tuning in this hybrid model configuration.

Feature Importance for Fake News Detection (ngram=(1, 1))
![image](https://github.com/user-attachments/assets/e39056ea-f8ae-4088-b9fb-059ab4d98342)

This horizontal bar plot displays the 20 words that the XGBoost model found most influential for classifying news articles. The length of each bar represents the word's "importance score," meaning words with longer bars had a greater impact on the model's predictions. Importantly, this score measures a word's overall impact, not which category (real or fake) it predicts.

The key takeaway is that the model has learned to identify distinct patterns. Words like **"gop"** (an abbreviation for the Republican party), **"pic"** (short for picture), and **"twitter"** are the most powerful predictors, suggesting that partisan language and informal, social-media-centric content are strong indicators of one class (likely fake news). Conversely, the model also relies on formal journalistic words like **"said"**, **"minister"**, and **"representatives"** as important signals, likely for identifying real news. This mix demonstrates that the model is successfully learning the different linguistic styles of both real and fake news articles.

Feature Importance for Fake News Detection (ngram=(1, 3))
![image](https://github.com/user-attachments/assets/0d8462eb-6eda-4df0-8d2b-bc72bfd1f64f)

This feature importance plot is incredibly revealing and provides a clear explanation for the model's behavior. The most striking finding is the overwhelming dominance of features related to Twitter, such as **"twitter com"**, **"president donald trump"**, **"pic twitter"**, and even the generic **"https"**. This strongly suggests that the model is not primarily learning from the semantic content or narrative of the news articles, but is instead taking a shortcut by identifying a structural pattern: the presence of embedded tweets or social media links is a powerful indicator of a document's class (likely fake news). The model has learned that articles containing raw text from Twitter are highly predictive, which is a common characteristic of fake news that often repackages or directly quotes social media posts. The inclusion of n-grams like **"president donald trump"** and **"president barack"** shows the model is correctly identifying key political figures, but the presence of generic words like **"just"**, **"said"**, and **"like"** further points to the model picking up on stylistic, informal, or conversational tones often found in opinionated or non-traditional news sources. Ultimately, this chart shows that the model is succeeding by acting more like a "format detector" than a "content analyzer," which, while effective on this dataset, highlights a potential vulnerability if it were to encounter real news that also embeds tweets.

  

 






























