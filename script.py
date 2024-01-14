from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



# Create the autoencoder model
def create_autoencoder(input_shape):
    input_layer = Input(shape=input_shape, name='input_layer')
    
    # Encoder
    encoded = Dense(500, activation='relu', name='encoded_layer')(input_layer)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu', name='latent_space')(encoded)

# Decoder
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(input_shape[0], activation='sigmoid')(decoded)
    
    
    ##
    
    # Decoder
    decoded = Dense(maxlen, activation='sigmoid', name='decoded_layer')(encoded)
    
    # Build and compile the autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    return autoencoder



loaded_data = np.load('your_file.npz')
tokenized = [loaded_data[key] for key in loaded_data.files]

loaded_data = np.load('tokenized_malacious.npz')
tokenized_malacious = [loaded_data[key] for key in loaded_data.files]

i=0
while i!=len(tokenized_malacious):
    if len(tokenized_malacious[i])<1000:
        del tokenized_malacious[i]
        i=0
    else: i=i+1

total_data=tokenized+tokenized_malacious[0:1200]


#maxlen = m
maxlen=1000000
padded_sequences = pad_sequences(total_data, padding='post', truncating='post', maxlen=maxlen)

autoencoder = create_autoencoder((maxlen,))

autoencoder.fit(padded_sequences, padded_sequences, epochs=10, batch_size=15, shuffle=True)

labels=np.ones(len(total_data))
labels[0:len(tokenized)]=0



features=autoencoder.predict(padded_sequences)
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42)
svm_classifier = SVC()
svm_classifier.fit(x_train, y_train)

y_pred = svm_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

forest=RandomForestClassifier(n_estimators=100)
forest.fit(x_train, y_train)

y_pred = forest.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

logistic_classifier = LogisticRegression()
logistic_classifier.fit(x_train, y_train)

y_pred = logistic_classifier.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

ann_classifier = Sequential([
    Dense(750, activation='relu', input_dim=100000),
    Dense(525, activation='relu'),
    Dense(256, activation='relu'),
    Dense(175, activation='relu'),# 32 is the size of the latent space in the autoencoder
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_classifier.fit(x_train, y_train, epochs=25, batch_size=10, shuffle=True)
accuracy = ann_classifier.evaluate(x_test, y_test)[1]
print(accuracy)

