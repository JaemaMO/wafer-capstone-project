import streamlit as st
from fastai.vision.all import *
import PIL
import numpy as np
import cv2
import torch
import os
import gdown
import timm

# 🚨 [핵심 해결] 학습할 때 사용했던 커스텀 평가 지표를 웹에도 똑같이 쥐어줍니다. (없으면 모델 로딩 실패)
recall_macro = Recall(average='macro')

# 1. 페이지 테마 및 타이틀 디자인 세팅
st.set_page_config(page_title="Wafer Defect Classifier", page_icon="🔍", layout="centered")

st.title("반도체 웨이퍼 결함 탐지 모델")
st.markdown("#### 🎓 명지대학교 산업경영공학과 캡스톤 디자인 프로젝트")
st.markdown("단순 증강과 디퓨전 증강으로 5000개로 증강한 모델입니다.")
st.divider()

# 2. 고속 캐싱 및 자동 다운로드 기반 모델 로드
@st.cache_resource
def load_model():
    model_path = 'wafer_export_model_mix.pkl'
    if not os.path.exists(model_path):
        with st.spinner('구글 드라이브에서 대용량 AI 모델 가중치를 최초 1회 다운로드 중입니다... (약 10~20초 소요)'):
            # 🚨 본인의 구글 드라이브 파일 ID (app.py 수정 시 덮어쓰기 주의!)
            file_id = '16W1Vh68cez9V7cDOhmofQtcBXCRdx-_t'
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
    return load_learner(model_path)

try:
    learn = load_model()
    base_classes = list(learn.dls.vocab)
    ui_classes = base_classes + ["모름 (Unknown)"]
except Exception as e:
    st.error(f"모델 파일 연동 실패: {e}")
    st.stop()

# [핵심 로직] AI 판단 근거 추적 Grad-CAM 함수
def get_gradcam_overlay(learner, pil_img, pred_idx):
    model = learner.model.eval()
    for param in model.parameters():
        param.requires_grad = True
        
    img_resized = pil_img.resize((112, 112))
    img_tensor = TensorImage(image2tensor(img_resized)).unsqueeze(0).float() / 255.0
    
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_(True)
    
    target_layer = model[0]
    activated_features = []
    gradients = []
    
    def forward_hook(module, input, output): 
        activated_features.append(output)
        output.register_hook(lambda grad: gradients.append(grad))
        
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    
    with torch.enable_grad():
        output = model(img_tensor)
        model.zero_grad()
        idx = int(pred_idx.item()) if hasattr(pred_idx, 'item') else int(pred_idx)
        loss = output[0, idx]
        loss.backward()
        
    handle_fwd.remove()
    
    if len(gradients) == 0 or len(activated_features) == 0:
        return np.array(img_resized)
        
    grads = gradients[0].cpu().data.numpy()[0]
    f_maps = activated_features[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(f_maps.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * f_maps[i]
        
    cam = np.maximum(cam, 0)
    if np.max(cam) == 0:
        cam = np.zeros(f_maps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += np.abs(w) * f_maps[i]
        cam = np.maximum(cam, 0)
        
    cam = cv2.resize(cam, (112, 112))
    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
        
    img_np = np.array(img_resized)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlayed = heatmap * 0.4 + img_np * 0.6
    return np.clip(overlayed, 0, 255).astype(np.uint8)

# 3. 렌더링 레이아웃 구조 정의
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 이미지 업로드")
    uploaded_file = st.file_uploader("분석할 웨이퍼 이미지 파일을 등록해 주세요.", type=['png', 'jpg', 'jpeg'])

with col2:
    st.subheader("2. 실제 정답 선택")
    st.markdown("채점 및 검증을 위해 이 웨이퍼의 실제 불량 종류를 선택해 주세요.")
    actual_label = st.selectbox("정답 선택 (Ground Truth)", ui_classes)

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.divider()
    img_show_col1, img_show_col2 = st.columns(2)
    with img_show_col1:
        st.image(img.resize((112,112)), caption="원본 이미지 (112px 변환)", width=250)
    
    if st.button("웨이퍼 결함 패턴 분석 시작", use_container_width=True):
        with st.spinner('AI가 결함 특징점 및 픽셀 그래디언트를 역추적 중입니다...'):
            pred, pred_idx, probs = learn.predict(img)
            confidence = probs[pred_idx] * 100
            gradcam_img = get_gradcam_overlay(learn, img, pred_idx)
            
            with img_show_col2:
                st.image(gradcam_img, caption="AI 판단 근거 (Grad-CAM)", width=250)
                st.caption("🔴 붉은색 영역: AI가 불량을 판정할 때 가장 집중해서 본 결함 패턴입니다.")
            
            st.subheader("📊 데이터 매칭 및 채점 결과")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="AI 분석 결과 (예측)", value=pred, delta=f"신뢰도: {confidence:.1f}%")
            with res_col2:
                st.metric(label="등록된 실제 정답", value=actual_label)
            
            if actual_label == "모름 (Unknown)":
                st.info(f"💡 실제 정답이 '모름' 상태이므로 채점(Pass/Fail)을 건너뜁니다. AI의 분석 결과는 [{pred}] 입니다.")
            elif pred == actual_label:
                st.success(f"🎉 정답매칭 성공! AI가 [{actual_label}] 결함 특징을 정확하게 검출해 냈습니다.")
                st.balloons()
            else:
                st.error(f"🚨 불일치 발생! AI는 [{pred}] 결함으로 판정했으나, 실제 기록은 [{actual_label}] 입니다.")
            
            with st.expander("결함 패턴별 매칭 확률 분포 요약"):
                for idx, cls in enumerate(learn.dls.vocab):
                    st.progress(float(probs[idx]), text=f"{cls}: {probs[idx]*100:.1f}%")
