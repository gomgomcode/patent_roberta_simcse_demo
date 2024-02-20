# Demo
https://huggingface.co/spaces/gomgomcode/patent_roberta_simcse_demo

# Model
https://huggingface.co/gomgomcode/material_patent_roberta_simcse

- 36만여건 특허데이터 본문 전처리 후 basemodel 학습(using roberta)
    (https://arxiv.org/abs/1907.11692)
- sentence embedding model 학습 (using SimCSE)
    (https://arxiv.org/abs/2104.08821)
- 제목, 본문, 요약데이터 활용하여 semi-supervised learning 진행

# Service
- 특허데이터베이스 사전 embedding
- faiss index 생성
- 검색어 임베딩 후 faiss index 활용 L2 distance 작은 top_k 결과 출력

# Config
---
title: Patent Roberta Simcse Demo
emoji: 🦀
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 4.19.1
app_file: app.py
pinned: false
license: mit
---