import numpy as np
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from collections import Counter
import Levenshtein as lev
import cv2
from pytesseract import *

# Generate synthetic data
data = ["hello", "world", "python", "keras", "model"]

# Create input and target sequences
input_texts = [word[:-1] for word in data]  # Remove the last character
target_texts = [word[1:] for word in data]   # Remove the first character

# Tokenize the characters
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(data)

input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# Pad sequences to have the same length
max_sequence_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')

# Convert target sequences to one-hot encoding
target_sequences = to_categorical(target_sequences, num_classes=len(tokenizer.word_index) + 1)

# Build the BLSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(input_sequences, target_sequences, epochs=100, verbose=2)

# Load the input image
#image_path = 'general_words/p84.jpg'
image_path = 'Fruits_img/blue.jpg'
#image_path='countries/c2.jpg'
# Minimum confidence value to filter weak text detection
min_conf = 0

# Load the input image and then convert it to RGB from BGR.
images = cv2.imread(image_path)
rgb = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

# Use Tesseract to localize each area of text in the input image
results = pytesseract.image_to_data(rgb, output_type=Output.DICT)

# Variable to accumulate the combined text
combined_text = ""

# Loop over each of the individual text localizations
for i in range(0, len(results["text"])):
    # Extract the bounding box coordinates of the text region from the current result
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]

    # Extract the OCR text itself along with the confidence of the text localization
    text = results["text"][i]
    conf = int(results["conf"][i])

    # Filter out weak confidence text localizations
    if conf > min_conf:
        # Display the confidence and text to the terminal
        print("Confidence: {}".format(conf))
        print("Text: {}".format(text))
        print("")

        # Append the recognized text to the combined text
        combined_text += text + " "

        # Strip out non-ASCII text so we can draw the text on the image
        text = "".join(text).strip()
        cv2.rectangle(images, (x, y), (x + w, y + h), (0, 0, 300), 2)
        cv2.putText(images, text, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)

# Function to predict missing character
def TextInference(w, W, n):
    # Calculate the prior probability of each word c in the dictionary W
    word_counts = Counter(W)
    total_words = len(W)
    prior_probabilities = {word: count / total_words for word, count in word_counts.items()}

    # Initialize the candidate recovery text to the input text w
    candidate_recovery_text = w

    for i in range(n + 1):  # Loop through edit distances from 0 to n
        # Find words in the dictionary W that have an edit distance of i from the input text w
        Ci = [word for word in W if lev.distance(w, word) == i]

        if Ci:
            # Calculate the posterior probability of each word in Ci based on prior probabilities
            posterior_probabilities = {word: prior_probabilities[word] for word in Ci}

            # Select the word with the highest posterior probability as the candidate recovery text
            candidate_recovery_text = max(posterior_probabilities, key=posterior_probabilities.get)
            return candidate_recovery_text  # Return the candidate recovery text

    # If no candidate was found within edit distance n, return the input text w
    return candidate_recovery_text

# Example usage:
print("Combined Text: {}".format(combined_text))
corrupt_text = combined_text.lower()
text_dictionary = ["Karnataka","apple", "banana", "cherry", "apricot", "apology", "appropriate", "blueberry", "green apple"
                  ,"science","university","research","institute","technology","master","storage","generate","australia","japan","brazil","india","china"]
edit_distance_threshold = 2

recovery_text = TextInference(corrupt_text, text_dictionary, edit_distance_threshold)
print("Candidate Recovery Text:", recovery_text)

# Create a blank image with larger size
combined_image = np.zeros((images.shape[0] + 100, images.shape[1], 3), dtype=np.uint8)

# Display the input image on the combined image
combined_image[0:images.shape[0], 0:images.shape[1]] = images

# Overlay the recovery text on the same window
cv2.putText(combined_image, recovery_text, (10, images.shape[0] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)

# Show the output image
cv2.imshow("Image", combined_image)
cv2.waitKey(0)
