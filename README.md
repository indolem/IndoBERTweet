# IndoBERTweet 🐦 :indonesia: 

## 1. Paper
Fajri Koto, Jey Han Lau, and Timothy Baldwin. [_IndoBERTweet: A Pretrained Language Model for Indonesian Twitter
with Effective Domain-Specific Vocabulary Initialization_](https://arxiv.org/pdf/2109.04607.pdf). 
In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (**EMNLP 2021**), Dominican Republic (virtual).

## 2. About

[IndoBERTweet](https://huggingface.co/indolem/indobertweet-base-uncased) is the first large-scale pretrained model for Indonesian Twitter
that is trained by extending a monolingually trained Indonesian BERT model with additive domain-specific vocabulary.

In this paper, we show that initializing domain-specific vocabulary with average-pooling of BERT subword embeddings is more efficient than pretraining from scratch, and more effective than initializing based on word2vec projections.

## 3. Pretraining Data

We crawl Indonesian tweets over a 1-year period using the official Twitter API, from December 2019 to December 2020, with 60 keywords covering 4 main topics: economy, health, education, and government. We obtain in total of **409M word tokens**, two times larger than the training data used to pretrain [IndoBERT](https://aclanthology.org/2020.coling-main.66.pdf). Due to Twitter policy, this pretraining data will not be released to public.

## 4. How to use

Load model and tokenizer (tested with transformers==3.5.1)
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
model = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")
```
**Preprocessing Steps:**
* lower-case all words 
* converting user mentions and URLs into @USER and HTTPURL, respectively
* translating emoticons into text using the [emoji package](https://pypi.org/project/emoji/).

## 5. Results over 7 Indonesian Twitter Datasets

<table>
  <col>
  <colgroup span="2"></colgroup>
  <colgroup span="2"></colgroup>
  <tr>
    <th rowspan="2">Models</td>
    <th colspan="2" scope="colgroup">Sentiment</th>
    <th colspan="1" scope="colgroup">Emotion</th>
    <th colspan="2" scope="colgroup">Hate Speech</th>
    <th colspan="2" scope="colgroup">NER</th>
    <th rowspan="2" scope="colgroup">Average</th>
  </tr>
  <tr>
    <th scope="col">IndoLEM</th>
    <th scope="col">SmSA</th>
    <th scope="col">EmoT</th>
    <th scope="col">HS1</th>
    <th scope="col">HS2</th>
    <th scope="col">Formal</th>
    <th scope="col">Informal</th>
  </tr>
  <tr>
    <td scope="row">mBERT</td>
    <td>76.6</td>
    <td>84.7</td>
    <td>67.5</td>
    <td>85.1</td>
    <td>75.1</td>
    <td>85.2</td>
    <td>83.2</td>
    <td>79.6</td>
  </tr>
  <tr>
    <td scope="row">malayBERT</td>
    <td>82.0</td>
    <td>84.1</td>
    <td>74.2</td>
    <td>85.0</td>
    <td>81.9</td>
    <td>81.9</td>
    <td>81.3</td>
    <td>81.5</td>
  </tr>
  <tr>
    <td scope="row">IndoBERT (Willie, et al., 2020)</td>
    <td>84.1</td>
    <td>88.7</td>
    <td>73.3</td>
    <td>86.8</td>
    <td>80.4</td>
    <td>86.3</td>
    <td>84.3</td>
    <td>83.4</td>
  </tr>
  <tr>
    <td scope="row">IndoBERT (Koto, et al., 2020)</td>
    <td>84.1</td>
    <td>87.9</td>
    <td>71.0</td>
    <td>86.4</td>
    <td>79.3</td>
    <td>88.0</td>
  <td><b>86.9</b></td>
    <td>83.4</td>
  </tr>
  <tr>
    <td scope="row">IndoBERTweet (1M steps from scratch)</td>
    <td>86.2</td>
    <td>90.4</td>
    <td>76.0</td>
  <td><b>88.8</b></td>
  <td><b>87.5</b></td>
  <td><b>88.1</b></td>
    <td>85.4</td>
    <td>86.1</td>
  </tr>
  <tr>
    <td scope="row">IndoBERT + Voc adaptation + 200k steps</td>
  <td><b>86.6</b></td>
  <td><b>92.7</b></td>
  <td><b>79.0</b></td>
    <td>88.4</td>
    <td>84.0</td>
    <td>87.7</td>
  <td><b>86.9</b></td>
  <td><b>86.5</b></td>
  </tr>
</table>
