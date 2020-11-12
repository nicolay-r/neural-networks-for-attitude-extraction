#!/bin/bash
data=../data
mkdir -p $data
curl http://rusvectores.org/static/models/rusvectores2/news_mystem_skipgram_1000_20_2015.bin.gz -o "$data/news_rusvectores2.bin.gz"
