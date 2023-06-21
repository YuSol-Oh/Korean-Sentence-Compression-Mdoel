# Korean-Sentence-Compression-Model
: Language Scoring with Morphological Analysis and Utilization of Perplexity

(한국어 문장 압축 모델 : 형태소 분석을 통한 언어 점수 부여 및 Perplexity 활용)

### ▶ 프로젝트 설계 배경

 국내 정보 탐색 방법을 보면, 대부분 모바일 기기를 통한 웹, 모바일 콘텐츠를 통해 정보를 획득한다. 그로 인해, 각 모바일 콘텐츠는 ‘작은 화면을 통해 사용자가 중요한 정보를 얼마나 빠르게 획득할 수 있는지’가 경쟁력이 되었다.
즉, 긴 정보를 작은 화면에 짧게 얼마나 잘 담는지가 중요해진 것이다.

 긴 정보를 짧게 줄이는 방식에는 모델이 문장의 의미를 파악하여 요약 문장을 생성하는 ‘문장 요약’ 방식과, 주어진 문장에서 중요하지 않은 정보를 제거하여 문장의 길이를 줄이는 ‘문장 압축’ 방식이 존재한다.
 
 문장 요약 모델의 경우, 생성 모델의 특성상, 문장이 논리적으로 맞다 판단되면 사실이 아님에도 사실처럼 문장을 생성해내는 hallucination 문제가 존재한다. 따라서 정보 요약 측면에서 정확도가 비교적 높은 ‘문장 압축’ 모델을 구현하고자 했다.

### ▶ 프로젝트 주요 내용

- 한국어는 영어와 달리 문장의 구성, 단어의 구성, 문법적 특성 상 언어 모델들을 적용하는 것에 어려움이 존재한다.
(ex. 한국어는 어순이 중요하지 않기 때문에 단어의 시퀀스(순서)에 대한 확률 추정이 효과가 없을 수 있음. 또한, 한국어는 영어와 다리 접사, 조사에 따라 문장성분 또는 품사가 달라질 수 있기 때문에 정확한 문장 분석이 어려움.)
- 따라서, 이러한 **한국어의 특성을 잘 반영하여 문장 요약을 하는 문장 압축 모델을 구현**하고자 한다.
- 또한, 입력하는 한국어 문장의 특성에 맞게 (ex. 뉴스 기사의 문장인지, 대화의 문장인지), 문장 속 단어들의 중요도를 분석 후 조절하여 납득 가능한 압축 문장을 생성하는 것이 목표이다.

### ▶ 문장 압축 모델 구현

➊ **데이터 수집**
   
- 압축 모델을 구현한 뒤, 문장이 잘 압축되었는지를 판단하기 위해 ‘압축 전 원본 문장’과 ‘압축 후 문장’ 쌍 데이터를 수집.
- ‘압축 전 원본 문장’은 뉴스 기사의 첫 번째 문장으로 하고, ‘압축 후 문장’은 뉴스 기사의 제목으로 정함.
- 모든 뉴스 기사의 첫 번째 문장과 기사 제목이 연관이 있는 것은 아니기 때문에, 그러한 경우 직접 압축 문장을 생성.

➋ **모델 구현**

1. `토큰화`
	- 입력한 문장을 토큰화하는 과정.
	- 기본적으로 문장을 어절 단위로 토큰화하지만, 뉴스 기사의 특성 상 따옴표(‘ ’) 사이의 단어들은 기사의 핵심 내용이거나 사건, 행사 등의 제목을 나타냄.
→ 따라서 따옴표와 따옴표를 포함하는 문장도 하나의 토큰으로 묶어서 토큰화를 진행함.
```python
def word_tokenizer(sentence):
    return sentence.split(' ')

  token = word_tokenizer(sentence)

  # 따옴표와 따옴표를 포함하는 문장을 하나의 토큰으로
  def process_quoted_words(tokens):
    processed_tokens = []
    quoted_word = ""
    in_quote = False

    for token in tokens:
      if "'" in token:
        if in_quote:
          quoted_word += " " + token
          processed_tokens.append(quoted_word)
          quoted_word = ""
          in_quote = False
        else :
          quoted_word = token
          in_quote = True
      else :
        if in_quote:
          quoted_word += " " + token
        else:
          processed_tokens.append(token)
    return processed_tokens
  token = process_quoted_words(token)
  ```

2. `언어정보점수 계산`
	- **언어정보점수** : 각 토큰들에 ‘문법적 특성’에 대한 점수를 부여한 것
	- 즉, 언어정보점수를 부여한다 = 중요한 정보를 갖고 있는 토큰을 판단하고, 문장 압축 과정에서 해당 토큰이 삭제되지 않도록 값을 조정하는 것.
	- 언어정보점수 부여는 문장의 형태소를 분석하여, 주요 형태소에 점수를 추가하는 방식으로 진행.
	- 기본적으로 문장의 주성분인 ‘주어, 목적어, 서술어’는 하나의 문장이 완성되기 위해서 반드시 필요한 성분들임. 
	→ 따라서 ‘주어’라고 판단할 수 있게 하는 ‘주격조사’, ‘목적어’라고 판단할 수 있게 하는 ‘목적격조사’, ‘서술어’라고 판단할 수 있게 하는 ‘동사’에 점수를 부여함.
	- 또한, 입력 문장이 뉴스 기사이라는 것을 고려하였을 때, 따옴표와 따옴표를 포함하는 문장의 중요도도 높다고 판단하여 해당 토큰에도 점수를 부여함.
```python
# 형태소 분석 후 언어정보점수 추가
  def add_linguistic_score(sentence):
    pos = komoran.pos(sentence)
    score = 0
    for p in pos:
      if p[1] == 'NNP': # 고유명사
        score += 0.0001
      elif p[1] == 'NNG': # 일반명사
        score += 0.0001
      elif p[1] == 'vv': # 동사 -> 서술어
        score += 0.03
      elif p[1] == 'SL' : # 외국어
        score += 0.0003
      elif p[1] == 'JKS' : # 주격 조사 -> 주어
        score += 0.01
      elif p[1] == 'JKO' : # 목적격 조사 -> 목적어
        score += 0.001
    quoted_words = re.findall(r"'(.*?)'", sentence)
    for quoted_word in quoted_words: # 따옴표 안의 단어들
      score += 0.000001
    return score
```

3. `압축 문장 후보군 생성 (n-gram 생성)`
	- 압축 문장 후보 : 원본 문장에서 특정 어절들을 삭제하여 만든 문장들
	- 이를 n-gram을 생성한다고 표현함.
	- 문자열에서 n개의 연속된 요소를 추출하여 삭제 헤서 압축 문장 후보들을 생성하고, 추출된 n개의 단어 자리에는 ‘[MASK]’라는 단어의 토큰으로 대체.
	- N-gram을 생성할 때에, 적어도 압축 문장 후보 문장들이 문장의 주성분인 주어, 목적어, 보어, 서술어는 최소한 남겨져야한다고 판단하여 압축 문장 후보의 단어가 4개 이상일 때 동안 압축 문장 후보 생성을 시행함.
```python
# n-gram
  compressed_candidates = []
  max_n = len(token) - 4 # 문장의 주성분인 주어,목적어,보어,서술어는 최소한 남겨두기 위해
  for n in range(1, max_n + 1):
    for i in range(len(token) - n + 1):
      compressed_tokens = token[:i] + token[i+n:]
      compressed_sentence = " ".join(compressed_tokens)
      # 언어정보점수 계산
      score = add_linguistic_score(compressed_sentence)
      # 압축 문장 후보에 언어정보점수 추가
      compressed_candidates.append((compressed_sentence,score,n))
```

4. `Perplexity 계산`
	- Perplexity란 **문장의 혼잡도**를 의미.
	- Perplexity 값이 작을수록 문장이 문맥적으로 이해 가능한 옳은 문자임을 의미하며, 값이 클수록 문장이 이해하기 어려워 혼잡함을 의미함.
	- Perplexity는, 문장을 KoBERT 모델이 이해할 수 있는 형태로 tokenize를 한 뒤, KoBERT가 [MASK] 토큰 위치에 대한 예측 값을 생성하고, softmax 함수로 예측 값의 확률 분포로 반환한 뒤, 문장에 대한 Generation probability의 역수의 기하평균을 구하는 방식으로 계산함.
```python
def calculate_perplexity_score(sentence):

    # KoBERT 모델이 이해할 수 있는 형태로 tokenize
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens) # tokenized word -> 정수 인덱스

    # [MASK] 토큰 위치 찾기
    mask_token_index = token_ids.index(tokenizer.mask_token_id)

    # 입려값과 출력값 생성
    input_ids = torch.tensor([token_ids])
    outputs = model(input_ids) # 예측값
    predictions = outputs[0]

    # [MASK] 토큰 위치에 대한 예측값 추출
    masked_predictions = predictions[0, mask_token_index]

    # softmax 함수를 이용하여 확률값을 확률 분포로 변환
    probs = torch.softmax(masked_predictions, dim = -1)

    # perplexity 계산
    perplexity = torch.exp(torch.mean(torch.log(probs)))

    return perplexity.item()
```

5. `최종 점수 계산 및 최종 압축 문장 선택`
	- 각 압축 문장 후보에 대해서 문장의 점수를 계산하여 부여하는 과정.
	- 각 문장 후보의 언어정보점수와 perplexity를 계산하고, 그 둘을 곱하여 ‘최종 점수’를 결정.
	- 이때, 두 점수를 곱하여 계산하는 이유는, perplexity 값은 0과 1사이의 수로 나타나며 점수가 낮을수록 옳은 문장이다. 따라서 언어정보점수 또한 0과 1 사이의 수로 나타내었고 중요도가 높은 형태소일수록 작은 값(0에 가까운 값)을 부여하였다. 이로 인해 두 점수가 곱해졌을 때에 문장의 문맥적으로 옳은 정도와 최종 점수 간에 상관관계가 만들어짐.
	- 각 문장에 대해 최종 점수 계산을 마친 뒤, 최종 점수가 낮은 순서대로 압축 문장 후보들을 정렬한 후, 최종 점수가 가장 낮은 압축 문장 후보를 최종 압축 문장으로 결정하며 문장 압축이 완료됨.
```python
  perplexity_scores = []
  compressed_candidates_with_score = []

  for n in range(1,max_n+1):
    for i in range(len(token) - n + 1 ):
      mask_idx = list(range(i, i+n))
      masked_tokens = list(token)
      for j in mask_idx:
        masked_tokens[j] = "[MASK]"
      masked_sentence = " " .join(masked_tokens)

      # perplexity score 계산
      perplexity_score = calculate_perplexity_score(masked_sentence)

      # 언어정보점수 가져오기
      linguistic_score = compressed_candidates[i][1]

      # 최종 점수 = perplexity_score + 언어정보점수
      final_score = perplexity_score * linguistic_score

      # perplexity_score 및 언어정보점수 저장
      perplexity_scores.append(perplexity_score)
      compressed_candidates_with_score.append((re.sub(r'\[MASK\]\s*', '', masked_sentence), final_score, n))

  # 최종 압축 문장 선택
  compressed_candidates_with_score_sorted = sorted(compressed_candidates_with_score, key=lambda x: x[1])

  final_compressed_sentence = re.sub(r'\[MASK\]\s*', '', compressed_candidates_with_score_sorted[0][0])
  selected_n = compressed_candidates_with_score_sorted[0][2]
```

➌ **모델 성능 평가**
   
- 모델의 성능 평가는 수집한 데이터의 ‘압축 전 원본 문장’을 구현한 문장 압축 모델을 통해 문장 압축을 실행하고, 이때 생성된 ‘최종 압축 문장’과 정답 문장인 데이터의 ‘압축 후 문장’을 비교하는 방식으로 진행.
- 정답 문장과 생성된 ‘최종 압축 문장’의 **코사인 유사도를 비교**하여 얼마나 유사한지를 평가함.
- 코사인 유사도 값이 크다면 문장 압축 모델의 성능이 좋은 것이며, 값이 작다면 성능이 좋지 않음을 의미.
- 모델의 성능평가는 데이터에서 random으로 50개의 문장에 대해 유사도를 비교하고, 이를 10번 반복하는 것을 한 번의 시행이라고 정한 뒤, 10번의 시행을 거치는 방식으로 진행하였음.
```python
def calculate_similarity(compressed_sentence, reference_sentence):

    # 형태소 분석 및 형태소 단위로 문장 분리
    compressed_morphemes = komoran.morphs(compressed_sentence)
    reference_morphemes = komoran.morphs(reference_sentence)

    # 모든 형태소 종류 추출
    all_morphemes = list(set(compressed_morphemes + reference_morphemes))

    # 형태소 빈도수 계산
    compressed_freq = {}
    reference_freq = {}

    for morpheme in all_morphemes:
        compressed_freq[morpheme] = compressed_morphemes.count(morpheme)
        reference_freq[morpheme] = reference_morphemes.count(morpheme)

    # 형태소 빈도수를 벡터로 변환
    compressed_vector = list(compressed_freq.values())
    reference_vector = list(reference_freq.values())

    # 코사인 유사도 계산
    similarity = cosine_similarity([compressed_vector], [reference_vector])[0][0]
    return similarity
```
