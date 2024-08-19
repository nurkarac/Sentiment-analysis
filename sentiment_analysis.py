import pandas as pd
import matplotlib.pyplot as plot
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings
from sklearn.model_selection import train_test_split, cross_val_score

filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: '%.2f' % x)
pd.set_option("display.width", 200)

# Dosya okutma
df = pd.read_excel("amazon.xlsx")
df.head()
df.info()

# Normalizing case folding (büyük harfleri küçük harf yap)
df["Review"] = df["Review"].str.lower()

# Noktalama işaretlerini kaldır
df["Review"] = df["Review"].str.replace(r'[^\w\s]', '', regex=True)

# Numaraları sil
df["Review"] = df["Review"].str.replace(r'\d', '', regex=True)

# Anlam taşımayan kelimeleri kaldır (Stopwords)
sw = stopwords.words('english')
df["Review"] = df["Review"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# Rarewords/ custom words az frekanslı kelimeleri çıkart
sil = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df["Review"] = df["Review"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sil))

# Lemmatization kelimeleri yalın hallerine dönüştürme
df["Review"] = df["Review"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df["Review"].head(10)

# Görselleştirme
tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plot.show()

# Wordcloud görselleştirme işlemi
text = " ".join(i for i in df.Review)

wordcloud = WordCloud(max_font_size=50,
                     max_words=100,
                     background_color="white").generate(text)
plot.figure()
plot.imshow(wordcloud, interpolation="bilinear")
plot.axis("off")
plot.show()

df.head()
sia = SentimentIntensityAnalyzer()
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))
df["Compound"] = df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])
df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["Sentiment_Label"] = df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df.groupby("Sentiment_Label")["Star"].mean()

# NaN değerleri temizleyin
df = df.dropna(subset=["Review", "Sentiment_Label"])

# Bağımlı ve bağımsız değişkenlerimizi belirleyerek datayı train-test olarak ayırınız.
# Test-Train
train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["Sentiment_Label"],
                                                    random_state=42)

# Verileri verebilmek için temsil şekillerini sayısala çevirmemiz gerek
# TF-IDF Word level
tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

# Modelleme (logistic regresyon)
# Logistic regresyon modelini kurarak train dataları ile fit ediniz
log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)

# Kurduğumuz model ile tahmin işlemi
y_pred = log_model.predict(x_test_tf_idf_word)
print(classification_report(y_pred, test_y))

# Cross-validation
cross_val_score_mean = cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=2).mean()
print(f'Cross-validation score mean: {cross_val_score_mean}')


random_review = pd.Series(df["Review"].sample(1).values)
yeni_yorum = CountVectorizer().fit(train_x).transform(random_review)
pred = log_model.predict(yeni_yorum)
print(f'Review: {random_review[0]} \n Prediction: {pred}')

# Modelleme (RandomForestClassifier)
rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
rf_cross_val_score_mean = cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()
print(f'Random Forest Cross-validation score mean: {rf_cross_val_score_mean}')
