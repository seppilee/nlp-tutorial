Word2vec을 제안한 T. Mikolov가 저자로 들어있으며 세줄로 요약하면 다음과 같다.

- Word embedding (Distributed vector represenatation of words)에는 다양한 방법이 있지만, 대부분의 방법들은 언어의 형태학적(Morpological)인 특성을 반영하지 못하고, 또 희소한 단어에 대해서는 Embedding이 되지 않음
- 본 연구에서는 단어를 Bag-of-Characters로 보고, 개별 단어가 아닌 n-gram의 Charaters를 Embedding함 (Skip-gram model 사용)
- 최종적으로 각 단어는 Embedding된 n-gram의 합으로 표현됨, 그 결과 빠르고 좋은 성능을 나타냈음


특히 기존연구의 한계점들에 대해서 설명하자면,

- “단어의 형태학적 특성을 반영하지 못했다”라는 것은 예를들어, teach와 teacher, teachers 세 단어는 의미적으로 유사한 단어임이 분명하다. 그런데 과거의 Word2Vec이나 Glove등과 같은 방법들은 이러한 단어들을 개별적으로 Embedding하기 때문에 셋의 Vector가 유사하게 구성되지 않는다는 점이다.
- “희소한 단어를 Embedding하기 어렵다”라는 것은 Word2Vec등과 같은 기존의 방법들은 Distribution hypothesis를 기반으로 학습하는 것이기 때문에, 출현횟수가 많은 단어에 대해서는 잘 Embedding이 되지만, 출현횟수가 적은 단어에 대해서는 제대로 Embedding이 되지 않는다는 점이다. (Machine learning의 용어로 설명하자면, Sample이 적은 단어에 대해서는 Underfitting이 되는 것으로 이해할 수 있겠다.)
