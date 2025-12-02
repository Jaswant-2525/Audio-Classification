**INTRODUCTION**

Audio classification is a machine learning task that involves automatically recognizing and categorizing sounds into predefined classes. Instead of processing images or text, the system takes an audio signal as input and analyzes it to identify patterns that distinguish different sounds. The raw audio is first preprocessed and transformed into meaningful representations such as Mel-Frequency Cepstral Coefficients (MFCCs), which capture the frequency and energy characteristics of the signal. These features are then used to train deep learning models like Convolutional Neural Networks (CNNs) which learn to map the features to specific labels. During prediction, the trained model can classify new audio samples into the correct category, such as “yes,” “no,” “up,” or “down.” The performance of the model is typically evaluated using metrics such as accuracy, precision, recall, and F1-score. Audio classification plays a vital role in applications like speech recognition, voice assistants, music analysis, and environmental sound detection.

**INPUT DATASET**
The inbuilt Speech Commands dataset in TensorFlow is provided through tensorflow_datasets (TFDS) under the name speech_commands. It is the official Google Speech Commands dataset and can be loaded directly without manually downloading files.

**Key Details:**

**Dataset Name in TFDS:** "speech_commands"

**Version Available:** Both v0.0.2 and v0.02 are available, depending on TFDS version.

**Audio Format:** WAV files, 1 second long, recorded at 16 kHz.

**Size:** About 1,00,500 audio clips.
1. **Training data size:** 85,511
2. **Validation data size:** 10,102
3. **Testing data size:** 4,890

**Classes:**

Contains 35 spoken words (e.g., yes, no, up, down, left, right, go, stop, on, off, zero–nine).

Includes background noise and unknown words.

**AFTER FILTERING**

The Speech Commands dataset originally has about 100,500 audio samples across 35 classes. By filtering, we keep only 4 out of 35 classes (which is 11% of the entire dataset).

The data is split by TensorFlow Datasets into:
1. Training set (85%)
2. Testing set (15%)

Approximate distribution (after filtering):
1. **Training samples per class:** 3,000–4,000
2. **Testing samples per class:** 400–500
3. **Total training size:** 12,440 samples
4. **Total testing size:** 1,655 samples

The number of samples in training and testing is balanced across classes, so each of the four classes (“yes”, “no”, “up”, “down”) has nearly equal representation.

**FEATURE EXTRACTION**

Feature extraction in audio classification is the process of converting raw audio signals (waveforms) into meaningful numerical representations that capture important sound characteristics. Since raw audio is just a sequence of amplitude values, it’s not directly suitable for machine learning models.

**MEL-FREQUENCY CEPSTRAL COEFFICIENT**

MFCC (Mel-Frequency Cepstral Coefficients) is a powerful feature representation used in audio and speech processing. It transforms raw audio into a compact set of coefficients that capture the most important characteristics of sound. MFCC represents audio data in terms of how humans perceive sound frequencies, focusing more on lower frequencies (where our ears are more sensitive) rather than treating all frequencies equally. This makes it especially effective for speech recognition.

**Features of MFCC:**

1. It is more compact and efficient than spectrograms.
2. It captures speech-relevant features rather than raw acoustic energy.
3. It reduces noise sensitivity, improving classification accuracy.
4. It is the standard choice in speech recognition tasks due to its alignment with human hearing.
