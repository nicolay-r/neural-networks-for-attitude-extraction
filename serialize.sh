#!/bin/bash
# Serialize sl
python2.7 -u run_serialization.py --ra-ver 1.1 --cv-count 3 --experiment sl --labels-count 2 --emb-filepath data/w2v/news_rusvectores2.bin.gz
python2.7 -u run_serialization.py --ra-ver 1.1 --cv-count 1 --experiment sl --labels-count 2 --emb-filepath data/w2v/news_rusvectores2.bin.gz
python2.7 -u run_serialization.py --ra-ver 1.1 --cv-count 3 --experiment sl --labels-count 3 --emb-filepath data/w2v/news_rusvectores2.bin.gz
python2.7 -u run_serialization.py --ra-ver 1.1 --cv-count 1 --experiment sl --labels-count 3 --emb-filepath data/w2v/news_rusvectores2.bin.gz

# Serialize sl+ds
python2.7 -u run_serialization.py --ra-ver 1.1 --cv-count 3 --experiment sl+ds --labels-count 2 --emb-filepath data/w2v/news_rusvectores2.bin.gz
python2.7 -u run_serialization.py --ra-ver 1.1 --cv-count 1 --experiment sl+ds --labels-count 2 --emb-filepath data/w2v/news_rusvectores2.bin.gz
python2.7 -u run_serialization.py --ra-ver 1.1 --cv-count 3 --experiment sl+ds --labels-count 3 --emb-filepath data/w2v/news_rusvectores2.bin.gz
python2.7 -u run_serialization.py --ra-ver 1.1 --cv-count 1 --experiment sl+ds --labels-count 3 --emb-filepath data/w2v/news_rusvectores2.bin.gz

# Serialize ds
python3.7 -u run_serialization.py --ra-ver 1.1 --cv-count 3 --experiment ds --labels-count 2 --emb-filepath data/w2v/news_rusvectores2.bin.gz
python2.7 -u run_serialization.py --ra-ver 1.1 --cv-count 3 --experiment ds --labels-count 3 --emb-filepath data/w2v/news_rusvectores2.bin.gz
