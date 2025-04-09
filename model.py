"""
모델 아키텍처 및 구현
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Input, 
    BatchNormalization, Reshape, Flatten, Lambda,
    MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.optimizers import Adam
import numpy as np

import config

def attention_module(x, num_heads=8):
    """
    주의 메커니즘 모듈
    """
    # 입력 형태 변환
    if len(x.shape) == 2:
        x = Reshape((1, x.shape[-1]))(x)
    
    # 자기 주의 메커니즘
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=64
    )(x, x)
    
    # 스킵 연결 및 정규화
    add_attention = Add()([x, attention_output])
    normalized = LayerNormalization()(add_attention)
    
    # 피드포워드 네트워크
    ffn = Dense(normalized.shape[-1]*2, activation='relu')(normalized)
    ffn = Dropout(0.1)(ffn)
    ffn = Dense(normalized.shape[-1])(ffn)
    
    output = Add()([normalized, ffn])
    output = LayerNormalization()(output)
    
    return output

def gradient_reversal_layer(x, lambda_=1.0):
    """
    그래디언트 반전 레이어 (도메인 적응용)
    """
    backward = lambda_ * -1.0
    
    @tf.custom_gradient
    def grad_reverse(x):
        def grad(dy):
            return backward * dy
        return x, grad
    
    return Lambda(grad_reverse)(x)

def create_model():
    """
    EfficientNetV2S 기반 감정 인식 모델 구축
    """
    # 입력 레이어
    inputs = Input(shape=(*config.IMAGE_SIZE, 3))
    
    # 기본 모델 선택
    if config.BASE_MODEL == "EfficientNetV2S":
        base_model = EfficientNetV2S(
            weights='imagenet', 
            include_top=False, 
            input_tensor=inputs
        )
    elif config.BASE_MODEL == "MobileNetV2":
        base_model = MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_tensor=inputs
        )
    else:
        raise ValueError(f"지원되지 않는 모델: {config.BASE_MODEL}")
    
    # 기본 모델 동결 (전이학습)
    if config.FREEZE_BASE_MODEL:
        for layer in base_model.layers:
            layer.trainable = False
    
    # 특징 추출
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # 주의 메커니즘 적용 (선택적)
    if config.USE_ATTENTION:
        x = Reshape((1, -1))(x)  # 주의 메커니즘을 위한 형태 변환
        x = attention_module(x)
        x = Flatten()(x)
    
    # 특징 임베딩
    x = Dense(1024, activation='relu', name='embedding')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # 감정 분류 출력
    emotion_predictions = Dense(config.NUM_CLASSES, activation='softmax', name='emotion')(x)
    
    # 기본 모델
    model = Model(inputs=inputs, outputs=emotion_predictions)
    
    if config.USE_DOMAIN_ADAPTATION:
        # 견종 불변성을 위한 도메인 적응 추가 (옵션)
        breed_features = gradient_reversal_layer(x)
        breed_predictions = Dense(10, activation='softmax', name='breed')(breed_features)  # 10은 예상 견종 수
        
        model = Model(inputs=inputs, outputs=[emotion_predictions, breed_predictions])
        
        # 컴파일: 도메인 적응 훈련
        model.compile(
            optimizer=Adam(learning_rate=config.INITIAL_LR),
            loss={
                'emotion': 'categorical_crossentropy',
                'breed': 'categorical_crossentropy'
            },
            loss_weights={
                'emotion': 1.0,
                'breed': 0.1  # 견종 분류는 보조 작업
            },
            metrics={
                'emotion': ['accuracy']
            }
        )
    else:
        # 기본 컴파일: 감정 분류만
        model.compile(
            optimizer=Adam(learning_rate=config.INITIAL_LR),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def unfreeze_model(model, fine_tuning_lr=1e-5):
    """
    미세조정을 위해 기본 모델 해제
    """
    # 모든 레이어 훈련 가능하게 설정
    for layer in model.layers:
        layer.trainable = True
    
    # 더 낮은 학습률로 재컴파일
    if config.USE_DOMAIN_ADAPTATION:
        model.compile(
            optimizer=Adam(learning_rate=fine_tuning_lr),
            loss={
                'emotion': 'categorical_crossentropy',
                'breed': 'categorical_crossentropy'
            },
            loss_weights={
                'emotion': 1.0,
                'breed': 0.1
            },
            metrics={
                'emotion': ['accuracy']
            }
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=fine_tuning_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def get_model_summary(model):
    """
    모델 요약 정보 반환
    """
    model.summary()
    return model