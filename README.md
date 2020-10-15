## Neural Networks in Sentiment Attitude Extraction 

This repository provides an application of neural for sentiment attitude extraction task.
The experiments provided within **RuSentRel** collection, as the latter yeilds of manually labeled news, with attitudes mentioned in it.
In order to increase the size of a train collection, it is possible to include **RuAttitudes** collection in training process.

## Dependencies

* AREKit == 0.20.5;
* Embeddings
    * RusVectores [[news-w2v-download]](http://rusvectores.org/static/models/rusvectores2/news_mystem_skipgram_1000_20_2015.bin.gz);
        
## How to run exepriment

Using `run_serialize_data.sh` in order to prepare data for a particular experiment, for a certain versions of 
RuAttiutdes and RuSentrel collections, appropriate input format, etc. 
(see **Model options** and **Evaluation formats** sections for details)

Using `run_train_classifier.sh` to run an experiment in following formats:
* **[RuSentRel](https://github.com/nicolay-r/RuSentRel)** collection -- supervised learning;
* **[RuAttitudes](https://github.com/nicolay-r/RuAttitudes)** collection -- distant supervision application for models pretraining;
* **RuSentRel with RuAttitudes** -- is a combined training, with an evaluation process within RuAttitudes collection.

## Model options

Supporting the following input formats:
* **Single instance** (Single context) 
* **Multi instance** (Multiple contexts, MaxPooling over multipe sentences by default) [TODO.  Provide ref]

## Evaluation formats

Evaluation formats:
* Fixed separation of RuSentRel 
* k-Fold cross validation.

## Bechmark results 

## References
> TODO. Provide references for such experiments, prior models.
