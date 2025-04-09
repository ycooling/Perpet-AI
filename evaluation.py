"""
모델 평가 및 분석 모듈
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config

def evaluate_model(model, validation_generator):
    """
    모델 성능 평가
    """
    print("모델 평가 중...")
    
    # 도메인 적응 모델 여부에 따른 평가
    if config.USE_DOMAIN_ADAPTATION:
        # 도메인 적응 모델은 감정 출력만 평가
        emotion_model = Model(inputs=model.input, outputs=model.get_layer('emotion').output)
        results = emotion_model.evaluate(validation_generator)
        print(f"감정 인식 정확도: {results[1]:.4f}")
    else:
        # 일반 모델 평가
        results = model.evaluate(validation_generator)
        print(f"모델 정확도: {results[1]:.4f}")
    
    return results

def get_predictions(model, validation_generator):
    """
    검증 데이터에 대한 예측 결과 얻기
    """
    # 도메인 적응 모델 여부에 따른 예측
    if config.USE_DOMAIN_ADAPTATION:
        emotion_model = Model(inputs=model.input, outputs=model.get_layer('emotion').output)
        predictions = emotion_model.predict(validation_generator)
    else:
        predictions = model.predict(validation_generator)
    
    # 진행 상황 표시
    print(f"검증 데이터 {len(validation_generator)} 배치에 대한 예측 완료")
    
    # 원-핫 인코딩된 실제 레이블
    true_labels = []
    steps = len(validation_generator)
    for i in range(steps):
        _, y = validation_generator[i]
        if isinstance(y, dict) and 'emotion' in y:
            true_labels.append(y['emotion'])
        else:
            true_labels.append(y)
    
    true_labels = np.concatenate(true_labels, axis=0)
    
    # 예측 결과를 클래스 인덱스로 변환
    pred_indices = np.argmax(predictions, axis=1)
    true_indices = np.argmax(true_labels, axis=1)
    
    return pred_indices, true_indices

def generate_classification_report(pred_indices, true_indices):
    """
    분류 성능 보고서 생성
    """
    # 클래스 이름 로드
    with open(os.path.join(config.OUTPUT_DIR, 'class_indices.json'), 'r') as f:
        class_indices = json.load(f)
    
    # 인덱스를 클래스 이름으로 매핑하기 위한 역변환
    class_names = {v: k for k, v in class_indices.items()}
    
    # 분류 보고서 생성
    report = classification_report(
        true_indices, 
        pred_indices, 
        target_names=[class_names[i] for i in range(len(class_names))],
        output_dict=True
    )
    
    # 결과를 DataFrame으로 변환하여 저장
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(config.OUTPUT_DIR, 'classification_report.csv'))
    
    # 혼동 행렬 생성
    cm = confusion_matrix(true_indices, pred_indices)
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=[class_names[i] for i in range(len(class_names))],
        yticklabels=[class_names[i] for i in range(len(class_names))]
    )
    plt.title('혼동 행렬')
    plt.ylabel('실제 클래스')
    plt.xlabel('예측 클래스')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    print("분류 보고서 및 혼동 행렬이 저장되었습니다.")
    
    return report_df

def generate_cam_visualization(model, img_path, class_idx, output_path=None):
    """
    클래스 활성화 맵(CAM) 시각화 생성
    """
    # 이미지 로드 및 전처리
    img = load_img(img_path, target_size=config.IMAGE_SIZE)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    
    # CAM 생성을 위한 모델 설정
    # 마지막 합성곱 레이어 찾기
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer = layer.name
            break
    
    if last_conv_layer is None:
        print("합성곱 레이어를 찾을 수 없습니다.")
        return None
    
    # 마지막 합성곱 레이어 및 예측 출력에 대한 그래디언트 모델 생성
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )
    
    # 그래디언트 계산
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, class_idx]
    
    # 기울기 추출
    grads = tape.gradient(loss, conv_outputs)
    
    # 풀링된 그래디언트
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 가중치 적용
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # 정규화
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # 히트맵 크기 조정
    heatmap = cv2.resize(heatmap, config.IMAGE_SIZE)
    
    # 히트맵을 RGB로 변환
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 원본 이미지와 히트맵 합성
    img_array = img_to_array(img)
    superimposed_img = heatmap * 0.4 + img_array
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # 결과 저장 또는 반환
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        print(f"CAM 시각화가 저장되었습니다: {output_path}")
    
    return superimposed_img

def evaluate_on_new_breeds(model, test_dir=None):
    """
    새로운 견종에 대한 모델 평가 (일반화 능력 테스트)
    """
    if test_dir is None:
        print("테스트 데이터 디렉토리가 지정되지 않았습니다.")
        return None
    
    # 테스트 제너레이터 생성
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # 모델 평가
    results = evaluate_model(model, test_generator)
    
    # 예측 및 분석
    pred_indices, true_indices = get_predictions(model, test_generator)
    report = generate_classification_report(pred_indices, true_indices)
    
    return results, report

def analyze_misclassifications(model, validation_generator, num_samples=10):
    """
    오분류 사례 분석
    """
    # 예측 가져오기
    if config.USE_DOMAIN_ADAPTATION:
        emotion_model = Model(inputs=model.input, outputs=model.get_layer('emotion').output)
        predictions = emotion_model.predict(validation_generator)
    else:
        predictions = model.predict(validation_generator)
    
    # 실제 레이블 가져오기
    true_labels = []
    image_batch_list = []
    steps = min(len(validation_generator), 10)  # 분석 시간 절약을 위해 10개 배치만 사용
    
    for i in range(steps):
        x_batch, y_batch = validation_generator[i]
        if isinstance(y_batch, dict) and 'emotion' in y_batch:
            y_batch = y_batch['emotion']
        true_labels.append(y_batch)
        image_batch_list.append(x_batch)
    
    x_all = np.concatenate(image_batch_list, axis=0)
    y_true = np.concatenate(true_labels, axis=0)
    y_pred = predictions[:len(y_true)]
    
    # 클래스 인덱스로 변환
    y_true_indices = np.argmax(y_true, axis=1)
    y_pred_indices = np.argmax(y_pred, axis=1)
    
    # 오분류 샘플 찾기
    misclassified_indices = np.where(y_true_indices != y_pred_indices)[0]
    
    # 클래스 이름 로드
    with open(os.path.join(config.OUTPUT_DIR, 'class_indices.json'), 'r') as f:
        class_indices = json.load(f)
    
    # 인덱스를 클래스 이름으로 매핑하기 위한 역변환
    class_names = {v: k for k, v in class_indices.items()}
    
    # 오분류 샘플 시각화
    if len(misclassified_indices) > 0:
        num_samples = min(num_samples, len(misclassified_indices))
        plt.figure(figsize=(20, 4 * num_samples))
        
        for i, idx in enumerate(misclassified_indices[:num_samples]):
            plt.subplot(num_samples, 2, 2*i+1)
            plt.imshow(x_all[idx])
            plt.title(f"실제: {class_names[y_true_indices[idx]]}\n예측: {class_names[y_pred_indices[idx]]}")
            plt.axis('off')
            
            # CAM 시각화
            cam_img = generate_cam_visualization(model, None, y_pred_indices[idx], None)
            if cam_img is not None:
                plt.subplot(num_samples, 2, 2*i+2)
                plt.imshow(cam_img)
                plt.title("클래스 활성화 맵")
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, 'misclassified_samples.png'))
        plt.close()
        
        print(f"오분류 사례 분석이 저장되었습니다: {os.path.join(config.OUTPUT_DIR, 'misclassified_samples.png')}")
        
    return len(misclassified_indices)