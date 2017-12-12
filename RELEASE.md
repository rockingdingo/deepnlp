# Release Log

## [0.1.8] - WIP
### Planned

## [0.1.7] - 2017-11-01
### Changed
tensorflow (=1.4) Support the lastest tensorflow 1.4 function changes
- Segment: Adding domain specific model: zh, zh_o2o, zh_entertainment
- POS/NER: Adding Bi-LSTM-CRF model: pos_model_bilstm_crf.py and ner_model_bilstm_crf.py
- NER: Provide Common Interface to Domain Specific models: zh_entertainment, zh_o2o, etc....
- Parsing: Adding dependency parser model, python implementation similar to Stanford NNDepParser
- textsum: textsum module are removed from this release, because the tensorflow Seq2Seq API changed to dynamic-rnn

## [0.1.6] - 2017-03-09
### Changed
tensorflow (=1.0) Support the latest tensorflow 1.0 function changes

### Added
- Restful API module: Calling deepnlp.org web NLP API access and usage
- textsum: implementing Seq2Seq-Attention model for automatic summarization, e.g. news headline generation
- textrank: textrank algorithm for extract and sort the important sentences

## [0.1.5] - 2016-11-15
### Changed
tensorflow (<=0.12.0)
### Added
- POS: Pre-trained English model using the Brown Corpus

## [0.1.4] - 2016-11-15
### Added
- Segment: Adding Chinese Segmentation models trained by CRF++ package
- POS: Including pre-trained Chinese models from China Daily corpus
- NER: Including pre-trained Chinese models from China Daily corpus

