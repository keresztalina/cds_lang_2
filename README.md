# Assignment 2 - Text classification benchmarks
This assignment is ***Part 2*** of the portfolio exam for ***Language Analytics S23***. The exam consists of 5 assignments in total (4 class assignments and 1 self-assigned project).

## Contribution
The initial assignment was created partially in collaboration with other students in the course, also making use of code provided as part of the course. The final code is my own. Several adjustments have been made since the initial hand-in.

Here is the link to the GitHub repository containing the code for this assignment: https://github.com/keresztalina/cds_lang_2

## 2.2. Assignment description by Ross
*(NB! This description has been edited for brevity. Find the full instructions in ```README_rdkm.md```.)*

This assignment is about using ```scikit-learn``` to train simple (binary) classification models on text data. For this assignment, we'll continue to use the Fake News Dataset that we've been working on in class.

For this exercise, you should write *two different scripts*. One script should train a logistic regression classifier on the data; the second script should train a neural network on the same dataset. Both scripts should do the following:

- Be executed from the command line
- Save the classification report to the folder called ```out```
- Save the trained models and vectorizers to the folder called ```models```

## Methods
The methods used in both scripts are near-identical. First, the data is loaded and split into train and test sets. Second, the data is vectorized: unigrams and bigrams are created, ignoring capitalization and removing the most common and most rare words, outputting the top 500 features. Third, a classifier is run on the vectorized data and predictions are made on the test data. Finally, the classifier, the model and the classification report are saved.

## Usage
### Prerequisites
This code was written and executed in the UCloud application's Coder Python interface (version 1.77.3, running Python version 3.9.2). UCloud provides virtual machines with a Linux-based operating system, therefore, the code has been optimized for Linux and may need adjustment for Windows and Mac.

### Installations
1. Clone this repository somewhere on your device. The data is already contained within the ```/cds_lang_2/in``` folder.
2. Open a terminal and navigate into the ```/cds_lang_2``` folder. Run the following lines in order to install the necessary packages and load the required language model:
        
        pip install --upgrade pip
        python3 -m pip install -r requirements.txt

### Run the script.
In order to run the script, make sure your current directory is still the ```/cds_lang_2``` folder. 

In order to run the logistic regression classifier, from command line, run:

        python3 src/LR.py
        
In order to run the neural net classifier, from command line, run:

        python3 src/MLP.py

The vectorizers and the models can be found in  ```/cds_lang_2/models```. The classification reports can be found in ```/cds_lang_2/out```.

## Discussion
Overall, both classifiers performed significantly better than chance on the fake vs real news dataset. The logistic regression classifier achieved a mean accuracy of 89%, performing with equal accuracy on the fake and the real headlines. The neural net classifier also achieved a mean accuracy of 89%, performing with equal accuracy on the fake and the real headlines.









