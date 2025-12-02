import librosa
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ds, info = tfds.load("speech_commands", split=['train', 'test'], with_info=True, as_supervised=True)

TARGET_CLASSES = ["yes", "no", "up", "down"]

train_ds = ds[0].filter(lambda audio, label: tf.reduce_any([label == info.features['label'].str2int(c) for c in TARGET_CLASSES]))
test_ds = ds[1].filter(lambda audio, label: tf.reduce_any([label == info.features['label'].str2int(c) for c in TARGET_CLASSES]))

print("Total training samples:", len(list(train_ds)))
print("Total testing samples:", len(list(test_ds)))

def extract_features_tf(audio, label, max_pad_len=40):
    # Convert to float32 in range [-1, 1]
    audio = tf.cast(audio, tf.float32) / 32768.0  # normalize int16 → float
    audio_np = audio.numpy()

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio_np, sr=16000, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    # Just return the integer label
    return mfccs, label

features, labels = [], []

for audio, label in tfds.as_numpy(train_ds):
    mfcc, lbl = extract_features_tf(audio, label)
    features.append(mfcc)
    labels.append(lbl)

for audio, label in tfds.as_numpy(test_ds):
    mfcc, lbl = extract_features_tf(audio, label)
    features.append(mfcc)
    labels.append(lbl)

X = np.array(features)
y = np.array(labels)

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = tf.keras.utils.to_categorical(y, num_classes=len(np.unique(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(40, 40, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),


    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

class_names = ["yes", "no", "up", "down"]

def predict_audio_tf(audio):
    # Convert int16 → float32
    audio = audio.numpy().astype(np.float32) / np.iinfo(np.int16).max

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
    max_pad_len = 40
    if mfccs.shape[1] < max_pad_len:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, max_pad_len - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    # Reshape for model
    mfccs = mfccs[..., np.newaxis]
    mfccs = np.expand_dims(mfccs, axis=0)

    # Predict
    prediction = model.predict(mfccs)
    predicted_index = np.argmax(prediction, axis=1)[0]

    return class_names[predicted_index]

for audio, label in test_ds.take(1):
    label_str = info.features['label'].int2str(label.numpy())

    if label_str not in class_names:
        print("Skipping sample with label:", label_str)
        continue

    true_label = label_str
    pred_label = predict_audio_tf(audio)

    print("True Label:", true_label)
    print("Predicted Label:", pred_label)

