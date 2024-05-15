import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer

num_words = 10000
(X_egitim, Y_egitim), (X_test, Y_test) = imdb.load_data(num_words=num_words)

maxlen = 130
X_egitim_pad = pad_sequences(X_egitim, maxlen=maxlen)
X_test_pad = pad_sequences(X_test, maxlen=maxlen)

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts([str(sentence) for sentence in X_egitim])

word_counts = tokenizer.word_counts
en_cok_tekrar_eden_kelimeler_kelime = [kelime for kelime, sayi in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]]

en_cok_tekrar_eden_kelimeler_str = ""
for i, kelime_id in enumerate(en_cok_tekrar_eden_kelimeler_kelime):
    kelime = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(int(kelime_id))]
    en_cok_tekrar_eden_kelimeler_str += f"{i + 1}. {kelime}: {word_counts[kelime_id]} kez\n"

en_cok_tekrar_eden_cumleler_str = ""
for i, cümle in enumerate(X_egitim[:10]):
    en_cok_tekrar_eden_cumleler_str += f"{i + 1}. Cümle: {' '.join(map(str, cümle))}\n"

print("En çok tekrar eden 10 kelime:")
print(en_cok_tekrar_eden_kelimeler_str)

print("En çok tekrar eden 10 cümle:")
print(en_cok_tekrar_eden_cumleler_str)

sentiment_model = Sequential()
sentiment_model.add(Embedding(num_words, 32))
sentiment_model.add(SimpleRNN(16, activation="relu"))
sentiment_model.add(Flatten())
sentiment_model.add(Dense(1, activation="sigmoid"))

print(sentiment_model.summary())
sentiment_model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

sentiment_history = sentiment_model.fit(X_egitim_pad, Y_egitim, validation_data=(X_test_pad, Y_test), epochs=5, batch_size=128, verbose=1)

sentiment_score = sentiment_model.evaluate(X_test_pad, Y_test)
print("Duygu Analizi Doğruluk: %", sentiment_score[1] * 100)

plt.figure()
plt.plot(sentiment_history.history["accuracy"], label="Eğitim")
plt.plot(sentiment_history.history["val_accuracy"], label="Test")
plt.title("Duygu Analizi Doğruluk")
plt.ylabel("Doğruluk")
plt.xlabel("Epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(sentiment_history.history["loss"], label="Eğitim")
plt.plot(sentiment_history.history["val_loss"], label="Test")
plt.title("Duygu Analizi Kayıp")
plt.ylabel("Kayıp")
plt.xlabel("Epochs")
plt.legend()
plt.show()
