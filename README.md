The Last Project: Analysis of Public Opinion Sentiment on Social Media Platform X Regarding the Free Nutritious Meal Program Using BERT-CNN

Crawling Twitter (Tweet Harvest v. 2.7.1)
keyword_1 = "makan bergizi gratis"
keyword_2 = "mbg"

## KONDISI AWAL

data: 172.009
detail: 70.075 [keyword_1] + 101.934 [keyword_2]
source: "dataset/raw/makan_bergizi_gratis" dan "dataset/raw/mbg"

## MERGE & VALIDATION

data: 118.340
detail: 41.933 [keyword_1] + 76.407 [keyword_2]
source: "dataset/merge/merged_mbg_full.csv"

## ELIMINATION

data: 53.669
detail: 28.142 [keyword_1] + 24.487 [keyword_2] + 1.0407 [keyword_2]

# SAMPLING DATA

data: 570 \* 12 = 6.840
data 1: seed batches label studio (200)
data 2: refill batches label studio (350)
data 3: refill batch val (20)
labeling: roberta label
source: "dataset/.csv"

data: 6.646
labeling: human annotation
source: "dataset/final/raw_sample_labeled.csv"

# FINAL DATA

pre-label
data: 6.646
source: "dataset/final/raw_sampling_mbg.csv"

# LABELING & VALIDATE

data: 6.642
source: "dataset/final/final_mbg_labeled.csv"
