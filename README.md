# Speaker-Verification-System-UsingResNet


go to dateset from here >>> [https://www.kaggle.com/datasets/kongaevans/speaker-recognition-dataset]

This repository implements a speaker verification system using a ResNet architecture built with Keras. It utilizes a training dataset of speakers' voices to learn speaker embeddings and classify whether a new voice sample belongs to a known speaker.

**Getting Started**

### Prerequisites

- Python 3.x (with TensorFlow 2.x or later)
- Keras
- pandas

You can install these dependencies using a package manager like `pip`:

```bash
pip install tensorflow keras pandas
```

**Data Preparation**

The code assumes you have a dataset of speaker audio files organized in a specific structure:

- `16000_pcm_speeches`: This is the root directory of your dataset.
- `audio`: This subfolder contains all the speaker audio files. Each speaker has its own subfolder within `audio`. 

**Code Breakdown**

The script follows these steps:

1. **Import Libraries:**
   - Imports necessary libraries for data manipulation, audio processing, and model building.

2. **Define Data Paths:**
   - Sets the paths to the dataset root directory and the audio subfolder.

3. **Load Audio Data:**
   - Gets a list of class names (speaker names) from the subfolders within the audio directory.
   - Iterates through each speaker's subfolder and creates lists of audio file paths and corresponding speaker labels (class indices).
   - Shuffles the audio paths and labels together to ensure random sampling during training and validation.

4. **Split Data:**
   - Splits the data into training and validation sets based on a predefined validation ratio (`VALID_SPLIT`).

5. **Create Datasets:**
   - Defines a function `paths_and_labels_to_dataset` that takes audio paths and labels as input and creates a TensorFlow `Dataset`.
   - Uses this function to create separate training and validation datasets.

6. **Preprocess Audio:**
   - Defines a function `path_to_audio` that reads and decodes a WAV audio file.
   - Defines a function `audio_to_fft` that converts the audio signal from the time domain to the frequency domain using Fast Fourier Transform (FFT).
   - Applies `audio_to_fft` to each audio sample within the training and validation datasets.

7. **Build ResNet Model:**
   - Defines a function `residual_block` that creates a residual block, a key component of ResNet architecture. 
   - Defines a function `build_model` that creates the entire ResNet model. The model takes Mel-frequency cepstral coefficients (MFCCs) as input (though the provided code uses frequency domain representation via FFT). It stacks several residual blocks with increasing filter sizes for feature extraction. The model uses global average pooling, flattening, and fully-connected layers with ReLU activation for classification. The final layer has a softmax activation with the number of units equal to the number of speaker classes.

8. **Compile Model (**This section is missing from the provided code snippet but can be added**):**
   - Compiles the model with an appropriate optimizer (e.g., Adam), loss function (e.g., categorical cross-entropy), and metrics (e.g., accuracy).

9. **Train Model (**This section is missing from the provided code snippet but can be added**):**
   - Trains the model on the training dataset for a specified number of epochs.
   - Evaluates the model's performance on the validation dataset using the compiled metrics.

**Note:**

- The provided code snippet focuses on data loading, preprocessing, and model definition. You'll need to implement the compilation and training sections for complete speaker verification.
- Consider using Mel-frequency cepstral coefficients (MFCCs) instead of raw frequency domain representation for better speaker recognition performance.

**Further Enhancements**

- Experiment with different ResNet configurations (e.g., number of residual blocks, filter sizes).
- Implement data augmentation techniques (e.g., noise injection, speed variations) to improve model robustness.
- Explore more advanced speaker embedding techniques like x-vectors.
- Use a pre-trained model on a larger speaker recognition dataset for transfer learning.

This code provides a starting point for building a speaker verification system with ResNet. You can customize and extend it based on your specific dataset and requirements.
