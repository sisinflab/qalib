# QALIB

A Question Answering LIBrary implemented in Python.

QALIB is a framework capable of mapping any questions provided by users with those belonging to a pre-trained FAQ set. It exploits an artificial neural network implemented in TensorFlow and some NLP tools provided by [nltk](https://www.nltk.org/) and [spacy](https://spacy.io/).

QALIB has both a command line interface for shell scripting and Python API for integration in other projects.

## Configuration

First of all, QALIB needs to be initialized by running:

```
qalib/setup.sh

python3 -m qalib.qalib --setup

```

They will automatically download the required dependencies and will configure the operating environment.

## Run

The following sums up the main functionalities of QALIB.

```
usage: qalib.py [-h] [-s] [-t <id> <file>] [-f <id> <question>] [-l <language>]

optional arguments:
  -h, --help            show this help message and exit
  -s, --setup           Run when using this framework for the first time.
  -t <id> <file>, --train <id> <file>
                        Add the FAQs related to the <id> item and contained
                        in <file>; each question has to be followed by its
                        answer; every question/answer couple has to be
                        separated from the others through a blank line.
  -f <id> <question>, --find <id> <question>
                        Search <question> among the FAQs of the <id> item.
  -l <language>, --language <language>
                        Select the language to use among the following: it for
                        Italian (default value) - en for English. 
```

To train the model on the selected FAQs, just run:

```
python3 -m qalib.qalib -t 1 qalib/faqs_example

```

Then it is possible to find the question asked by the user among them:

```
python3 -m qalib.qalib -f 1 "Quali impianti posso utilizzare con le centrali MyNice?"

```

QALIB output is a serialized JSON object containing the index of the found question according to the sorted list of FAQs in the file provided at the training phase.

```
{"result": "success", "index": "1"}

```

## API usage

The following example shows how to integrate QALIB in a Python project.

```
# Import QALIB
from qalib import qalib

# Set the id and the FAQ file path
id = "1"
file = "qalib/faqs_example"

# Train the model
qalib.add_newFAQs(id, file)
qalib.findFAQ(id, language="it", training=True)

# Prediction
question = "Quali impianti posso utilizzare con le centrali MyNice?"
qalib.findFAQ(id, user_question=question, language="it", training=False)

```

## Credits

This library has been developed by Angelo Schiavone while working at [SisInf Lab](http://sisinflab.poliba.it) under the supervision of Tommaso Di Noia.

## Contacts

Tommaso Di Noia, tommaso [dot] dinoia [at] poliba [dot] it

Angelo Schiavone, angelo [dot] schiavone [at] poliba [dot] it

