import img as img
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from PIL import Image
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM
from nltk.corpus import stopwords
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense, SpatialDropout1D
from keras.layers import Flatten, Conv1D, LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

df=pd.read_csv('./deneme.csv',encoding= 'utf-16')
df= df.drop([df.index[13] ,df.index[45],df.index[32],df.index[34],df.index[54],df.index[57],df.index[67],df.index[96],df.index[99],df.index[109],df.index[114],df.index[136],df.index[138],df.index[130],df.index[139],df.index[142],df.index[155],df.index[156]])
df.reset_index(inplace=True, col_level=1)
df=df.drop(columns='index', axis=1)
df_new= df['Complaint']


if __name__ == '__main__':

    ZEMBEREK_PATH: str = join('..', 'bin', 'zemberek-full.jar')

    startJVM(
        getDefaultJVMPath(),
        '-ea',
        f'-Djava.class.path={ZEMBEREK_PATH}',
        convertStrings=False
    )

    TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
    TurkishSentenceNormalizer: JClass = JClass(
        'zemberek.normalization.TurkishSentenceNormalizer'
    )
    Paths: JClass = JClass('java.nio.file.Paths')

    normalizer = TurkishSentenceNormalizer(
        TurkishMorphology.createWithDefaults(),
        Paths.get(
            join('..', 'data')
        ),
        Paths.get(
            join('..', 'data', 'lm.2gram.slm')
        )
    )

    nlp_list = []

    for i in range(len(df_new)):
        print(df_new[i])
        text = str(normalizer.normalize(JString(df_new[i])))
        nlp_list.append(text)

        print("---------------")
        print(nlp_list[i])
        print("----------------")

    df = df.drop(["Complaint"], axis=1)
    df.insert(1, "Complaint", nlp_list)

    shutdownJVM()


stop_words = set(stopwords.words('turkish'))

new_stopwords = ['ankara', 'büyükşehir', 'belediyesi', 'bir', ',', '.', 'mavi', 'rağmen', 'yok', '?', 'kadar', '*', 'hiçbir', 'masa', ':', '!', 'masaya', 'masayı', 'olan' , 'olarak']
new_stopwords_list = stop_words.union(new_stopwords)

print(new_stopwords_list)


def clean_text(Complaint):

    Complaint = ' '.join(
        word for word in Complaint.split() if word not in new_stopwords_list)  # remove stopwors from text
    return Complaint


df['Complaint'] = df['Complaint'].apply(clean_text)

df = df.drop(["Headline", "Date", "View count"], axis=1)

# df.to_csv(r'.\corrected_mavimasa.csv', encoding='utf-16', sep='|')

df2 = pd.read_csv(r'.\tag.csv', encoding='ISO-8859-9')

df.insert(1, "Tag", df2, True)

sns.countplot(x='Tag', data=df)
plt.show()


total = float(len(df)) # one person per row
#ax = sns.barplot(x="class", hue="who", data=titanic)
ax=sns.countplot(x='Tag', data=df, order = df['Tag'].value_counts().index) # for Seaborn version 0.7 and more
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center")
plt.show()



# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 5000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 100
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Complaint'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


X = tokenizer.texts_to_sequences(df['Complaint'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(df['Tag'].values)
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 32

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)])

accr = model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))



plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()


def new_comment(text):
    new_complaint = [text]
    seq = tokenizer.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    labels = ['altyapı', 'aski','ulaşım','diğer','servisler','personel']
    return print(pred, labels[np.argmax(pred)])

#<---KONSOLDA DENEME İÇİN ---> new_comment('DENEME CÜMLESİ')

all_strings = []
complaint = df['Complaint']

for i in complaint:
    all_strings += i.split()

counter = Counter(all_strings)
most_occur = counter.most_common(20)
most_common_str = ""

for i in range(len(most_occur)):
    most_common_str += " " + most_occur[i][0]

# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(most_common_str)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

df_ulasim = df.loc[df.Tag == 'ulaşım']

def word_cloud(tag):
    df_tag = df.loc[df.Tag == tag]
    all_strings = []
    complaint = df_tag['Complaint']

    for i in complaint:
        all_strings += i.split()

    counter = Counter(all_strings)
    most_occur = counter.most_common(50)
    most_common_str = ""

    for j in range(len(most_occur)):
        most_common_str += " " + most_occur[j][0]

    # Create and generate a word cloud image:
    wordcloud = WordCloud().generate(most_common_str)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    hat_mask = np.array(Image.open('./anahtar.png'))
    wordcloud = WordCloud(mask=hat_mask,background_color="white").generate(most_common_str)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.show()








