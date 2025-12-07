
# 필수 라이브러리 설치
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium[box2d] stable-baselines3 pandas numpy matplotlib yfinance ta shap
```

# 모델 학습 및 저장 
```
python3 scripts/calculate_model.py
```

# 결과 분석 및 시각화 (XAI Analysis)
```
python3 scripts/get_analysis.py
```
