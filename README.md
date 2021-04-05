## Neural Networks Applications in Sentiment Attitude Extraction 

This repository is an application for neural-networks of AREkit framework, devoted to sentiment attitude extraction task.

It Provides applications for:
* Data serialization;
* Training neural networks (see model list);

## Models List

> TODO. Provide the related list from AREkit==0.20.5 contributional part.

## Dependencies

* AREKit == 0.20.5

## Installation

> TODO. Clarify information from the install dir. Remove the latter and move everything here.

## Applications

### Networks Input Data Serialization

The experiments provided within **RuSentRel** collection, as the latter yeilds of manually labeled news, with attitudes mentioned in it.
In order to increase the size of a train collection, it is possible to include **RuAttitudes** collection in training process.

### Networks Training and Evaluation

* Embeddings
    * RusVectores [[news-w2v-download]](http://rusvectores.org/static/models/rusvectores2/news_mystem_skipgram_1000_20_2015.bin.gz);

Using `run_serialize_data.sh` in order to prepare data for a particular experiment, for a certain versions of 
RuAttiutdes and RuSentrel collections, appropriate input format, etc. 
(see **Model options** and **Evaluation formats** sections for details)

Using `run_train_classifier.sh` to run an experiment in following formats:
* **[RuSentRel](https://github.com/nicolay-r/RuSentRel)** collection -- supervised learning;
* **[RuAttitudes](https://github.com/nicolay-r/RuAttitudes)** collection -- distant supervision application for models pretraining;
* **RuSentRel with RuAttitudes** -- is a combined training, with an evaluation process within RuAttitudes collection.

Input formats:
* **Single instance** (Single context) 
* **Multi instance** (Multiple contexts, MaxPooling over multiple sentences by default) [TODO.  Provide ref]
* **Attention-based Multi instance** [TODO. Provide ref]

### Evaluation formats

* Fixed separation of RuSentRel 
* k-Fold cross validation.

> TODO. Provide script details `run_serialization.py`, `run_training.py`

## References
> TODO. Provide references for such experiments, prior models.
