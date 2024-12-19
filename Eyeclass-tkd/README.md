이 프로젝트는 눈 감음과 하품을 탐지하는 간단한 코드를 제공합니다.

### 설치 및 설정
```
git clone https://github.com/Hmiru/EYE_class_for_inha.git
cd EYE_class_for_inha
python main.py
```

### 주요 기능
**눈 감음 탐지**
- EAR 기반으로 눈 감김 여부를 추적하고 감지된 눈 감음 이벤트를 기록합니다.
- thresh.ear=0.15, consecutive frame=20

**하품 탐지**
-  MAR 기반으로하품 여부를 감지하고 하품 횟수를 기록합니다.
-  thresh.mar=0.7, consecutive frame=10
