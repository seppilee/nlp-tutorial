FastText는 Facebook에서 만든 word representation과 sentence classification의 효율적인 학습을 위한 라이브러리입니다.

https://fasttext.cc/

## Cheatsheet

**Word representation learning**
   - In order to learn word vectors do:

>$ ./fasttext skipgram -input data.txt -output model

**Obtaining word vectors**
   - Print word vectors for a text file queries.txt containing words.

>$ ./fasttext print-word-vectors model.bin < queries.txt

**Text classification**
   - In order to train a text classifier do:

>$ ./fasttext supervised -input train.txt -output model

   - Once the model was trained, you can evaluate it by computing the precision and recall at k (P@k and R@k) on a test set using:

>$ ./fasttext test model.bin test.txt 1

   - In order to obtain the k most likely labels for a piece of text, use:

>$ ./fasttext predict model.bin test.txt k

   - In order to obtain the k most likely labels and their associated probabilities for a piece of text, use:

>$ ./fasttext predict-prob model.bin test.txt k

   - If you want to compute vector representations of sentences or paragraphs, please use:

>$ ./fasttext print-sentence-vectors model.bin < text.txt

**Quantization**
   - In order to create a .ftz file with a smaller memory footprint do:

>$ ./fasttext quantize -output model

   - All other commands such as test also work with this model

>$ ./fasttext test model.ftz test.txt

**Autotune**
   - activate hyperparameter optimization with -autotune-validation argument:

>$ ./fasttext supervised -input train.txt -output model -autotune-validation valid.txt

  - timeout (in seconds):

>$ ./fasttext supervised -input train.txt -output model -autotune-validation valid.txt -autotune-duration 600

  - Constrain the final model size:

>$ ./fasttext supervised -input train.txt -output model -autotune-validation valid.txt -autotune-modelsize 2M
```

