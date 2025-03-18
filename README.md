물론이죠! 한글로 작성된 README.md 파일 템플릿을 제공해 드리겠습니다.

---

# **🎬 IMDB 리뷰 감성 분석을 위한 MobileBERT 미세 조정**

MobileBERT 로고  
**PyTorch와 HuggingFace Transformers를 사용하여 IMDB 영화 리뷰 데이터셋에 대해 MobileBERT를 미세 조정합니다.**

---

## **📌 프로젝트 개요**
이 프로젝트는 경량화되고 효율적인 트랜스포머 모델인 **MobileBERT**를 IMDB 영화 리뷰 데이터셋에 미세 조정하여 감성을 **긍정** 또는 **부정**으로 분류하는 방법을 보여줍니다. 목표는 높은 정확도를 달성하면서도 엣지 디바이스에 배포하기에 적합한 효율성을 유지하는 것입니다.

---

## **🛠️ 주요 기능**
- **MobileBERT 미세 조정**: HuggingFace의 `transformers` 라이브러리를 활용한 효율적인 학습
- **IMDB 데이터셋**: IMDB 영화 리뷰 데이터셋 샘플 사용
- **GPU 가속**: CUDA 지원 디바이스에 최적화
- **커스터마이징 가능**: 다른 텍스트 분류 작업에 쉽게 적용 가능
- **모델 저장**: 미세 조정된 모델을 나중에 사용할 수 있도록 저장

---

## **📂 파일 구조**
```
├── mobilebert_finetune.py       # 미세 조정을 위한 주 Python 스크립트
├── imdb_reviews_sample.csv      # 샘플 데이터셋 (IMDB 리뷰)
├── README.md                    # 프로젝트 문서
└── mobilebert_finetuned_imdb/   # 미세 조정된 모델이 저장되는 디렉토리
```

---

## **🚀 시작하기**

### 1️⃣ **저장소 클론**
```bash
git clone https://github.com/yourusername/mobilebert-finetune-imdb.git
cd mobilebert-finetune-imdb
```

### 2️⃣ **의존성 설치**
Python 3.8 이상이 설치되어 있는지 확인하세요. 그 다음, 필요한 라이브러리를 설치하세요:
```bash
pip install torch transformers scikit-learn pandas tqdm
```

### 3️⃣ **데이터셋 준비**
데이터셋 파일(`imdb_reviews_sample.csv`)을 프로젝트 디렉토리에 위치시키세요. CSV 파일은 다음 구조를 가져야 합니다:
| Text                          | Sentiment |
|-------------------------------|-----------|
| "이 영화는 환상적이었어요!"    | 1         |
| "이 영화를 정말 싫어했어요."   | 0         |

- `Text`: 리뷰 텍스트
- `Sentiment`: 1은 긍정, 0은 부정

### 4️⃣ **스크립트 실행**
데이터셋에 대해 MobileBERT를 미세 조정하세요:
```bash
python mobilebert_finetune.py
```

---

## **📊 결과**

학습 후, 터미널에서 에폭별 결과를 확인할 수 있습니다:
```
에폭 1: 학습 손실: 0.3456, 학습 정확도: 0.8765, 검증 정확도: 0.8543
에폭 2: 학습 손실: 0.2345, 학습 정확도: 0.9123, 검증 정확도: 0.8921
...
```

미세 조정된 모델은 `mobilebert_finetuned_imdb/` 디렉토리에 저장됩니다.

---

## **📈 성능**
- **모델**: MobileBERT (영어 소문자 텍스트로 사전 학습됨)
- **데이터셋**: IMDB 리뷰 (샘플)
- **정확도**: 검증 데이터에서 약 90% (샘플 크기에 따라 다름)

---

## **🖼️ 사용 예시**
미세 조정 후, 모델을 로드하여 감성을 예측할 수 있습니다:
```python
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer

# 미세 조정된 모델과 토크나이저 로드
model = MobileBertForSequenceClassification.from_pretrained("mobilebert_finetuned_imdb")
tokenizer = MobileBertTokenizer.from_pretrained("mobilebert_finetuned_imdb")

# 새로운 리뷰의 감성 예측
review = "이 영화를 정말 좋아했어요!"
inputs = tokenizer(review, return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax().item()
print("감성:", "긍정" if prediction == 1 else "부정")
```
