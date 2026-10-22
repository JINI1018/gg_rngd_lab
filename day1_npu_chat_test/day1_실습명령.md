# 2인 1조 실습 미션 명령
# 실습1 furiosa-ai/Qwen3-8B-FP8
## std1 명령
FuriosaAI의 NPU(RNGD)에서 Qwen3-8B 모델을 로딩하고, 외부 프로그램이 API로 추론을 요청할 수 있도록 서버를 실행함.
```
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```
----
|구성 요소|의미|
|---|---|
|furiosa-llm|	FuriosaAI NPU에서 LLM 추론을 실행하는 소프트웨어|
|serve|	모델을 로딩하고 API 서버를 실행하는 명령|
|furiosa-ai/|	Hugging Face의 모델 저장소 네임스페이스|
|Qwen3-8B|	약 80억 개의 파라미터를 가진 Qwen3 언어모델|
|FP8|	모델 가중치에 8비트 부동소수점 표현을 사용하는 버전|
----
* --reasoning-parser qwen3: 모델의 추론 과정과 최종 답변을 구분해 처리한다.
* --default-chat-template-kwargs: 기본적으로 Thinking 모드를 비활성화한다.

## std2

```
watch -n 1 furiosa-smi status
```

## 추론 수행 테스트
```
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "furiosa-ai/Qwen3-8B-FP8",
    "messages": [
      {
        "role": "user",
        "content": "NPU와 GPU의 차이를 설명해줘."
      }
    ],
    "max_tokens": 128,
    "chat_template_kwargs": {
      "enable_thinking": false
    }
}' | python3 -m json.tool

```


# 실습2 furiosa-ai/Qwen3-VL-4B-Instruct
## std2
```
furiosa-llm serve furiosa-ai/Qwen3-VL-4B-Instruct \
  --port 8000 \
  --allowed-local-media-path /home/std2/images \
  --image-limit-per-prompt 4
```
* --port 8000 :	API 서버 포트 지정
* --allowed-local-media-path :	지정 디렉터리의 로컬 이미지 접근 허용
* --image-limit-per-prompt 4 :	요청당 이미지 최대 4개로 제한

## std1
```
watch -n 1 furiosa-smi status
```

## 추론 수행 테스트
```
npu_chat_qwen3-8b.py
npu_chat_qwen3-VL-4b.py
```