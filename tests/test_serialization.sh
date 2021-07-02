#!/bin/bash
cd .. && python run_serialization.py \
    --cv-count 3 --frames-version v2_0 \
    --experiment rsr+ra --labels-count 3 --ra-ver dbg\
    --emb-filepath data/news_rusvectores2.bin.gz \
    --entity-fmt rus-simple --balance-samples True