# Code for paper Tackling the ADReSS challenge: a multi-view approach to the automated recognition of Alzheimer’s dementia #


## Installation, documentation ##

Published results were produced in Python 3 programming environment on Linux Mint 18 Cinnamon operating system. Instructions for installation assume the usage of PyPI package manager.<br/>

Install dependencies if needed: pip install -r requirements.txt

Download OpenSmile library source and unpack it in the root folder: https://www.audeering.com/download/opensmile-2-3-0-tar-gz/

### Instructions: ###

Generate ADR audio features:<br/>
```
python feature_generation/generate_adr_features.py --input_path pathToAudioFilesFolder --opensmile_path pathToOpensmileLibrary --output_path opensmileOutputfolder --output_tsv_path  pathToFolderForAllTsvFeatures
```

Generate other audio features:<br/>
```
python feature_generation/generate_audio_features.py --input_path pathToAudioFilesFolder --output_tsv_path pathToFolderForAllTsvFeatures
```

Convert audio transcripts to textual features:<br/>
```
python feature_generation/parse_transcripts.py --input_path pathToTranscriptsFolder --output_path pathToFolderForAllTsvFeatures
```

Grid search across all feature combinations and all learners:<br/>
```
python cross_validation.py --input_path pathToFolderForAllTsvFeatures --output_model_path pathToFolderWithSavedTrainedModels --meta_path pathToTwoFilesWithMMSELabels --feature_set all_combs
```

To train regression models instead of classification models add a --regression flag.<br/>

To use the trained model on the test set, first generate features for a test set without labels (same commands as for train set, just add --train flag) and then call:

```
python predict.py --input_path pathToFolderForAllTsvFeatures --model_path pathToSavedTrainedModel --result_path pathToFolderThatShouldContainResults
```

* [Knowledge Technologies Department](http://kt.ijs.si), Jožef Stefan Institute, Ljubljana
