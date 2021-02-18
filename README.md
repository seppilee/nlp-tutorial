## nlp-tutorial

<p align="center"> <img width="200" src="https://pytorchnlp.readthedocs.io/en/latest/_static/logo.svg" /></p>


파이토치를 이용한 자연어처리 실습 튜토리얼입니다. <br>
모든 모델은 짧은 코드로 컨셉을 이해하기 위해 알기 쉽게 작성되었습니다. 


#### 1. NLP 전처리

- Colab - [형태소분석](https://colab.research.google.com/drive/15L7szv7yqYmWT0e2rnhNhwkETh4gy6QC#scrollTo=7Tvsl2qSIxSo)


#### 2. 단어수준의 임베딩 모델

- 2-1. [NNLM(Neural Network Language Model)](1-1.NNLM) - **Predict Next Word**
  - 요약 - [NNLM](https://github.com/seppilee/nlp-tutorial/blob/main/NNLM.md)
  - Colab - 
  - Paper -  [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
   
- 2-2. [Word2Vec(Skip-gram)](1-2.Word2Vec) - **Embedding Words and Show Graph**
  - 요약 -[Word2vec](https://github.com/seppilee/nlp-tutorial/blob/main/W2V.md)
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - Colab - [Word2Vec.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec_Skipgram(Softmax).ipynb)
- 2-3. [FastText(Application Level)](1-3.FastText) - **word embedding**
  - 요약 - [FastText](https://github.com/seppilee/nlp-tutorial/blob/main/fasttext.md)
  - Paper - [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)
  - Colab - [FastText.ipynb](https://colab.research.google.com/drive/1-bXceLS06-sc1paQV0GKutPg1Qu7t5Fs#scrollTo=y1yDPCjVsO6x)



#### 2. CNN(Convolutional Neural Network)

- 2-1. [TextCNN](2-1.TextCNN) - **Binary Sentiment Classification**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
  - [TextCNN.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/2-1.TextCNN/TextCNN.ipynb)



#### 3. RNN(Recurrent Neural Network)

- 3-1. [TextRNN](3-1.TextRNN) - **Predict Next Step**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
  - Colab - [TextRNN.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-1.TextRNN/TextRNN.ipynb)
- 3-2. [TextLSTM](https://github.com/graykode/nlp-tutorial/tree/master/3-2.TextLSTM) - **Autocomplete**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - Colab - [TextLSTM.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-2.TextLSTM/TextLSTM.ipynb)
- 3-3. [Bi-LSTM](3-3.Bi-LSTM) - **Predict Next Word in Long Sentence**
  - Colab - [Bi_LSTM.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-3.Bi-LSTM/Bi_LSTM.ipynb)



#### 4. Attention Mechanism

- 4-1. [Seq2Seq](4-1.Seq2Seq) - **Change Word**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
  - Colab - [Seq2Seq.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-1.Seq2Seq/Seq2Seq.ipynb)
- 4-2. [Seq2Seq with Attention](4-2.Seq2Seq(Attention)) - **Translate**
  - Paper - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
  - Colab - [Seq2Seq(Attention).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention).ipynb)
- 4-3. [Bi-LSTM with Attention](4-3.Bi-LSTM(Attention)) - **Binary Sentiment Classification**
  - Colab - [Bi_LSTM(Attention).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-3.Bi-LSTM(Attention)/Bi_LSTM(Attention).ipynb)
- 4.4 opennmt
  - Colab - [OpenNMT Pytorch version tutorial](https://colab.research.google.com/drive/1QJdbj4MWTPS5pSWdm8OcZSl6Q75khRPy)

#### 5. 트랜스포머 기반의 언어 모델

- 5-1.  [The Transformer](5-1.Transformer) - **Translate**
  - Paper - [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
  - Colab - [Transformer.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer.ipynb), [Transformer(Greedy_decoder).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer(Greedy_decoder).ipynb)
- 5-2. [BERT](5-2.BERT) - **Classification Next Sentence & Predict Masked Tokens**
  - Paper - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)
  - Colab - [BERT.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT.ipynb)
- 5.3 [KoBERT]
  - Colab - [SKT Kobert](https://colab.research.google.com/drive/1lyTSeXO2tfvXbFeu_lqEBJ-EgMOSHHvZ)

## Dependencies

- Python 3.5+
- Pytorch 1.0.0+

