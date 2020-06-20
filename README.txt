A recipe for doing inflection-set KWS.

Dependencies:
- Software: Kaldi, F4DE, a Python >= 3.7 virtualenv in tools/venv37
- Data: Babel, Wall Street Journal, hub4 Spanish, voxforge French and Russian,
  Unimorph for ground truth paradigms.

uam/asr1/run.py is the master script for running experiments and should be the
starting point for understanding how the code works. In particular, the code after the line
`__name__ == "__main__":` can be used as a starting point for getting a sense
of how to run experiments. it can be used to prepare the data, train models and run KWS and
evaluation. It's in the state it was when running some of the final
experiments that gave results in the paper, but will likely need to be adjusted
to run on a new machine. Note that the inflection hypotheses from RNN-DTL were
generated using a different codebase. If you're interested in using those
particular inflections, reach out to Garrett Nicolai (gnicola2@jhu.edu).

Layout explanation:
- raw/ contains raw data before any of our preprocessing.
- tools/ contains relevant prerequisite tools, including F4DE and the Python
virtual environment. It would be better if it also contained kaldi, but
currently kaldi is at the root level of this directory.
- uam/ contains the universal acoustic model recipe and the KWS run.py script.
- explore_data.py is used to assess overlap between Babel transcriptions and
  unimorph data for preparing test sets.
