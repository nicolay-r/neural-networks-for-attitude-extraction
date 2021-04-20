## Neural Networks Applications in Sentiment Attitude Extraction 

This repository is an application for neural-networks of AREkit framework, devoted 
to sentiment **attitude extraction** task [[initial-paper]](https://arxiv.org/pdf/1808.08932.pdf), applied for a document **contexts**:

![](docs/task-intro.png)
> Figure: Example of a context with attitudes mentioned in
it; named entities **«Russia»** and **«NATO»** have the negative attitude towards each other with additional indication of other
named entities.

It provides applications for:
* [Data serialization](#application-1-data-serialization);
* [Training neural networks](#application-2-training) for the following [models list](#models-list).

## Models List

* **Aspect-based Attentive encoders**:
    - Multilayer Perceptron (MLP)
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/attention/architectures/mlp.py) /
        [[github:nicolay-r]](https://github.com/nicolay-r/mlp-attention);
* **Self-based Attentive encoders**:
    - P. Zhou et. al.
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/attention/architectures/self_p_zhou.py) /
        [[github:SeoSangwoo]](https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction);
    - Z. Yang et. al.
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/attention/architectures/self_z_yang.py) /
        [[github:ilivans]](https://github.com/ilivans/tf-rnn-attention);
* **Single Sentence Based Architectures**:
    - CNN
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/context/architectures/cnn.py) /
        [[github:roomylee]](https://github.com/roomylee/cnn-relation-extraction);
    - CNN + Aspect-based MLP Attention
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/context/architectures/base/att_cnn_base.py);
    - PCNN
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/context/architectures/pcnn.py) /
        [[github:nicolay-r]](https://github.com/nicolay-r/sentiment-pcnn);
    - PCNN + Aspect-based MLP Attention
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/context/architectures/base/att_pcnn_base.py);
    - RNN (LSTM/GRU/RNN)
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/context/architectures/rnn.py) /
        [[github:roomylee]](https://github.com/roomylee/rnn-text-classification-tf);
    - IAN (frames based)
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/context/architectures/ian_frames.py) /
        [[github:lpq29743]](https://github.com/lpq29743/IAN);
    - RCNN (BiLSTM + CNN)
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/context/architectures/rcnn.py) /
        [[github:roomylee]](https://github.com/roomylee/rcnn-text-classification);
    - RCNN + Self Attention
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/context/architectures/rcnn_self.py);
    - BiLSTM
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/context/architectures/bilstm.py) /
        [[github:roomylee]](https://github.com/roomylee/rnn-text-classification-tf);
    - Bi-LSTM + Aspect-based MLP Attention 
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/context/architectures/base/att_bilstm_base.py)
    - Bi-LSTM + Self Attention
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/context/architectures/self_att_bilstm.py) /
        [[github:roomylee]](https://github.com/roomylee/self-attentive-emb-tf);
    - RCNN + Self Attention
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/context/architectures/att_self_rcnn.py);
* **Multi Sentence Based Encoders Architectures**:
    - Self Attentive 
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/multi/architectures/att_self.py);
    - Max Pooling
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/multi/architectures/max_pooling.py) /
        [[paper]](https://pdfs.semanticscholar.org/8731/369a707046f3f8dd463d1fd107de31d40a24.pdf);
    - Single MLP
        [[code]](https://github.com/nicolay-r/AREkit/tree/0.20.5-rc/contrib/networks/multi/architectures/base/base_single_mlp.py);

## Dependencies

* AREKit == 0.20.5

## Installation

AREkit repository:
```shell script
# Clone repository in local folder of the currect project. 
git clone -b 0.20.5-rc git@github.com:nicolay-r/AREkit.git ../arekit
# Install dependencies.
pip install -r arekit/requirements.txt
```

### Prepare the data

We utilize RusVectores `news-2015` embedding:
```shell script
mkdir -p data
curl http://rusvectores.org/static/models/rusvectores2/news_mystem_skipgram_1000_20_2015.bin.gz -o "data/news_rusvectores2.bin.gz"
```

## Application #1. Data Serialization

Using `run_serialization.sh` in order to prepare data for a particular experiment:

```shell script
python run_serialization.py 
    --cv-count 3 --frames-version v2_0 
    --experiment rsr+ra --labels-count 3 
    --emb-filepath data/news_rusvectores2.bin.gz 
    --entity-fmt rus-simple --balance-samples True
```

## Application #2. Training

Using `run_train_classifier.sh` to run an experiment.

```shell script
CUDA_VISIBLE_DEVICES=0 python run_training.py --do-eval 
    --bags-per-minibatch 32 --dropout-keep-prob 0.80 --cv-count 3 
    --labels-count 3 --experiment rsr --model-input-type ctx 
    --model-name cnn --test-every-k-epoch 5 --learning-rate 0.1 
    --balanced-input True --train-acc-limit 0.99  --epochs 100
```

## Script Arguments Manual

Common flags:
* `--experiment` -- набор данных обучения моделей:
    * `rsr` -- коллекция [RuSentRel](https://github.com/nicolay-r/RuSentRel) для обучения с учителем;
    * `ra` -- коллекция [RuAttitudes](https://github.com/nicolay-r/RuAttitudes) для предобучения;
    * `rsr+ra` -- combined RuSentRel и RuAttitudes.
* `--cv\_count` -- формат разбиения набора данных;
    * 1 -- использование фиксированного формата разбиения;
    * k -- использование кросс-валидационного разбиения на $k$-частей;
* `--frames_versions` -- версия коллекции RuSentiFrames{}:
    * `v2.0` -- коллекция фреймов RuSentiFrames-2.0;
* `--ra_ver` -- версия коллекции RuAttitudes (если используется):
    * `v1_2` -- коллекция \ruattitudesVersion{1.0};
    * `v2_0_base` -- коллекция \raBase{};
    * `v2_0_large` -- коллекция \raLarge{};
    * `v2_0_base_neut` -- коллекция \raBase{}-Neut;
    * `v2_0_large_neut` -- коллекция \raLarge{}-Neut;
    
Training specific flags:
* `--model_name` -- название используемого кодировщика в общей модели нейронной сети [[list]](#models-list);
* `--do_eval` -- флаг, указывает на выполнение оценки модели в процессе обучения;
* `--bags_per_minibatch` -- количество мешков в мини-партии;
* `--balanced_input` -- флаг, указывает на использование сбалансированной коллекции в обучении модели;
* `--emb-filepath` -- путь к предобученной \wordtovec{} модели векторных представлений слов;
* `--entity-fmt` -- тип форматирования термов сущностей в контексте.
    * `rus-simple`  -- использование русскоязычных строк-масок: объект, субъект, сущость;
    * `sharp-simple` -- использование следующих масок: \#O (для объектов), \#S (для субъектов), \#E (для остальных сущностей контекста); такой формат представления используется в языковых моделях;
* `--balance-samples` -- флаг включения/отключения балансировки коллекции контекстов по классам;
