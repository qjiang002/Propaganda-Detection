# Propaganda Detection

This is a solution for datathon [Hack the News Datathon Case – Propaganda Detection](https://www.datasciencesociety.net/hack-news-datathon-case-propaganda-detection/) task 1 and task 2.<br>
The solutoin report can be found here. [Propaganda Detection in Social Media](https://drive.google.com/file/d/171RE_w6U6cYdCLNQ0PIjuh8UIZfoAEqG/view?usp=sharing)

### Environment setup
python3<br>
tensorflow >= 1.11.0<br>
install package `spacy`<br>
install `en_core_web_sm` by `python -m spacy download en_core_web_sm`<br>
install `vader_lexicon` by `nltk.download('vader_lexicon')`<br>
Download [bert](https://github.com/google-research/bert) into folder `bert`<br>
Download [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) into folder `checkpoint`<br>

### Datasets
The original datasets can be downloaded from the datathon website.<br>
In `data` folder, the data has been processed for BERT input.<br>

### Machine Learning models
`task1_ML.ipynb`: ML models include SVM, Logistic Regression, Random Forest and KNN for task 1, using full articles and article summaries.<br>
`task2_ML.ipynb`: ML models include SVM, Logistic Regression, Random Forest and KNN for task 2, using text sentences and supplementary features (named entities and sentimental polarities).<br>

### BERT-based models
* `BERT_classifier.py`: BERT classification<br>
Example:
```
python BERT_classifier.py --data_dir=data/task_2 --bert_config_file=checkpoint/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=checkpoint/uncased_L-12_H-768_A-12/bert_model.ckpt --vocab_file=checkpoint/uncased_L-12_H-768_A-12/vocab.txt --output_dir=./output/BERT_classifier --max_seq_length 128 --do_train --do_eval --do_predict 2>&1 | tee output/BERT_classifier/training.log
```

* `BERT_post_matching.py`: integrate supplementary features (named entities and sentimental polarities) into BERT by post-matching. Details are in the solution report.<br>
Training parameters:
```
--data_dir=data/task_2
--post_matching = mean/concat (default: mean)
--use_ner = True/False (default: True)
--use_polarity = True/False (default: True)
```
Example:
```
python BERT_post_matching.py --data_dir=data/task_2 --bert_config_file=checkpoint/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=checkpoint/uncased_L-12_H-768_A-12/bert_model.ckpt --vocab_file=checkpoint/uncased_L-12_H-768_A-12/vocab.txt --output_dir=./output/BERT_post_matching --max_seq_length 128 --do_train --do_eval --do_predict --post_matching=mean --use_ner --use_polarity 2>&1 | tee output/BERT_post_matching/training.log
```

* `BERT_ner_embedding.py`: integrate named entity features into BERT by input embedding. Polarity features are optionally integrated by post-matching. Details are in the solution report.<br>
Training parameters:
```
--data_dir=data/task_2
--post_matching = mean/concat (default: mean)
--use_polarity = True/False (default: True)
```
Example:
```
python BERT_ner_embedding.py --data_dir=data/task_2 --bert_config_file=checkpoint/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=checkpoint/uncased_L-12_H-768_A-12/bert_model.ckpt --vocab_file=checkpoint/uncased_L-12_H-768_A-12/vocab.txt --output_dir=./output/BERT_ner_embedding --max_seq_length 128 --do_train --do_eval --do_predict --post_matching=mean --use_polarity 2>&1 | tee output/BERT_ner_embedding/training.log
```

* `BERT_multitask.py`: multi-task training tasks include propaganda text classification, NER sequence labelling and sentimental polarity text classification. Details are in the solution report.<br>
Training parameters:
```
--data_dir=data/task_2_ner
--polarity_threshold = 0.4 (The threshold of the absolute value of polarity compound score. default: 0.4)
--use_ner = True/False (default: True)
--use_polarity = True/False (default: True)
```
Example:
```
python BERT_multitask.py --data_dir=data/task_2_ner --bert_config_file=checkpoint/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=checkpoint/uncased_L-12_H-768_A-12/bert_model.ckpt --vocab_file=checkpoint/uncased_L-12_H-768_A-12/vocab.txt --output_dir=./output/BERT_multitask --max_seq_length 128 --do_train --do_eval --do_predict --use_ner --use_polarity --polarity_threshold=0.4 2>&1 | tee output/BERT_multitask/training.log
```
