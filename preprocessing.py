"""
데이터 로딩 및 전처리 모듈
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import albumentations as A
import pandas as pd
from sklearn.model_selection import train_test_split
import json

import config

def get_class_names():
    """
    데이터셋에서 클래스명(감정 라벨) 추출
    """
    class_names = [d for d in os.listdir(config.DATA_DIR) 
                   if os.path.isdir(os.path.join(config.DATA_DIR, d))]
    return sorted(class_names)

def create_data_generators():
    """
    기본 데이터 제너레이터 생성
    """
    # 기본 데이터 증강 설정
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=config.VALIDATION_SPLIT,
        **config.AUGMENTATION
    )
    
    # 검증용 제너레이터 (증강 없음)
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=config.VALIDATION_SPLIT
    )
    
    # 훈련 데이터 로드
    train_generator = datagen.flow_from_directory(
        config.DATA_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # 검증 데이터 로드
    validation_generator = validation_datagen.flow_from_directory(
        config.DATA_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # 클래스 정보 저장
    class_indices = train_generator.class_indices
    print(f"클래스 정보: {class_indices}")
    
    # JSON으로 클래스 정보 저장
    with open(os.path.join(config.OUTPUT_DIR, 'class_indices.json'), 'w') as f:
        json.dump(class_indices, f)
    
    # 클래스 수 설정
    config.NUM_CLASSES = len(class_indices)
    
    return train_generator, validation_generator

def preprocess_dog_face(img_path, target_size=config.IMAGE_SIZE):
    """
    개 얼굴 중심 전처리 (옵션)
    """
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 여기에 개 얼굴 감지 로직 추가 가능
        # (이 부분은 옵션이므로 간단하게 유지)
        
        # 리사이즈 및 정규화
        img = cv2.resize(img, target_size)
        return img / 255.0
    except Exception as e:
        print(f"이미지 처리 중 오류: {e}")
        # 오류 시 검정색 이미지 반환
        return np.zeros((*target_size, 3))

def create_advanced_augmentation():
    """
    Albumentations 기반 고급 증강 파이프라인 생성
    """
    return A.Compose([
        A.OneOf([
            A.RandomContrast(limit=0.3),
            A.RandomBrightness(limit=0.3),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.4),
        A.OneOf([
            A.GridDistortion(distort_limit=0.1),
            A.ElasticTransform(alpha=1, sigma=20),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3)
    ])

class MixupGenerator(tf.keras.utils.Sequence):
    """
    Mixup 데이터 증강을 위한 제너레이터
    """
    def __init__(self, generator, alpha=0.2):
        self.generator = generator
        self.alpha = alpha
        
    def __len__(self):
        return len(self.generator)
        
    def __getitem__(self, idx):
        x_batch, y_batch = self.generator[idx]
        batch_size = x_batch.shape[0]
        
        # Mixup 가중치 생성
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        lam = np.maximum(lam, 1 - lam)  # 대칭성 보장
        lam_reshaped = lam.reshape(batch_size, 1, 1, 1)
        
        # 인덱스 섞기
        index_array = np.random.permutation(batch_size)
        
        # Mixup 적용
        mixed_x = lam_reshaped * x_batch + (1 - lam_reshaped) * x_batch[index_array]
        mixed_y = lam.reshape(batch_size, 1) * y_batch + (1 - lam.reshape(batch_size, 1)) * y_batch[index_array]
        
        return mixed_x, mixed_y

def apply_mixup(train_generator):
    """
    훈련 제너레이터에 Mixup 적용
    """
    if config.USE_ADVANCED_AUGMENTATION:
        return MixupGenerator(train_generator, alpha=config.MIXUP_ALPHA)
    return train_generator

def get_breed_invariant_pairs(generator, num_pairs=1000):
    """
    견종 불변성 학습을 위한 이미지 쌍 생성 (옵션 기능)
    """
    # 구현은 실제 데이터셋 구조에 따라 달라질 수 있음
    # 여기서는 개념만 제시
    pass