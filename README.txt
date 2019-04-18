Things that need to be done:
- Install prerequisites: Kaldi, F4DE, a Python >= 3.7 virtualenv in tools/venv37
- Acquire raw data. For speech this includes Babel, Wall Street Journal (wsj),
  hub4 spanish, voxforge French and Russian. For text this means Unimorph data
  for our ground-truth paradigms.
- Run the UAM (universal acoustic model) recipe in full:
	- Activate virtualenv
	cd uam/asr1; python run.py

Layout explanation:
- raw/ contains raw data before any of our preprocessing.
- tools/ contains relevant prerequisite tools, including F4DE and the Python
virtual environment. It would be better if it also contained kaldi, but
currently kaldi is at the top level.
- uam/ contains the universal acoustic model recipe.
- explore_data.py is used to assess overlap between Babel transcriptions and
  unimorph data for preparing test sets.
