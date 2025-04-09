"""
모델 훈련 및 학습 프로세스
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
import time
import numpy as np
import matplotlib.pyplot as plt

import config
import preprocessing
import model

def create_callbacks():
    """
    훈련용 콜백 함수 생성
    """
    # 모델 체크포인트
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(config.MODEL_DIR, 'model_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # 조기 종료
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # 학습률 감소
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.LR_REDUCTION_FACTOR,
        patience=config.LR_REDUCTION_PATIENCE,
        min_lr=1e-6,
        verbose=1
    )
    
    # 텐서보드 로깅
    tensorboard = TensorBoard(
        log_dir=os.path.join(config.LOG_DIR, f'run_{int(time.time())}'),
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # CSV 로깅
    csv_logger = CSVLogger(
        os.path.join(config.LOG_DIR, 'training_log.csv'),
        append=True
    )
    
    return [checkpoint, early_stopping, reduce_lr, tensorboard, csv_logger]

def train_model():
    """
    단계적 훈련 프로세스 실행
    """
    print("데이터 제너레이터 준비 중...")
    train_generator, validation_generator = preprocessing.create_data_generators()
    
    # Mixup 적용 (선택적)
    if config.USE_ADVANCED_AUGMENTATION:
        print("Mixup 데이터 증강 적용 중...")
        train_generator = preprocessing.apply_mixup(train_generator)
    
    print("모델 구축 중...")
    dog_emotion_model = model.create_model()
    model.get_model_summary(dog_emotion_model)
    
    print("1단계: 기본 모델 동결 상태로 훈련 시작...")
    callbacks = create_callbacks()
    
    # 도메인 적응 훈련인 경우
    if config.USE_DOMAIN_ADAPTATION:
        # 견종 레이블이 필요하지만 없으므로 더미 데이터 생성
        # 실제 구현에서는 견종 정보를 활용해야 함
        class BreedGenerator(tf.keras.utils.Sequence):
            def __init__(self, generator):
                self.generator = generator
                
            def __len__(self):
                return len(self.generator)
                
            def __getitem__(self, idx):
                x_batch, y_batch = self.generator[idx]
                batch_size = x_batch.shape[0]
                # 10개 견종 클래스에 대한 더미 라벨 (랜덤)
                breed_labels = np.random.rand(batch_size, 10)
                breed_labels = breed_labels / breed_labels.sum(axis=1, keepdims=True)
                return x_batch, {'emotion': y_batch, 'breed': breed_labels}
        
        wrapped_train_generator = BreedGenerator(train_generator)
        wrapped_validation_generator = BreedGenerator(validation_generator)
        
        history = dog_emotion_model.fit(
            wrapped_train_generator,
            validation_data=wrapped_validation_generator,
            epochs=config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = dog_emotion_model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
    
    # 첫 번째 훈련 후 모델 저장
    dog_emotion_model.save(os.path.join(config.MODEL_DIR, 'model_stage1.keras'))
    
    # 미세조정 단계
    print("2단계: 미세조정 훈련 시작...")
    dog_emotion_model = model.unfreeze_model(dog_emotion_model, config.FINE_TUNING_LR)
    
    # 새 콜백으로 미세조정
    fine_tuning_callbacks = create_callbacks()
    
    if config.USE_DOMAIN_ADAPTATION:
        fine_tuning_history = dog_emotion_model.fit(
            wrapped_train_generator,
            validation_data=wrapped_validation_generator,
            epochs=config.FINE_TUNING_EPOCHS,
            callbacks=fine_tuning_callbacks,
            verbose=1
        )
    else:
        fine_tuning_history = dog_emotion_model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=config.FINE_TUNING_EPOCHS,
            callbacks=fine_tuning_callbacks,
            verbose=1
        )
    
    # 최종 모델 저장
    dog_emotion_model.save(os.path.join(config.MODEL_DIR, 'model_final.keras'))
    
    # 훈련 결과 시각화
    plot_training_history(history, fine_tuning_history)
    
    return dog_emotion_model

def plot_training_history(history, fine_tuning_history=None):
    """
    훈련 과정 시각화
    """
    plt.figure(figsize=(12, 4))
    
    # 정확도 그래프
    plt.subplot(1, 2, 1)
    if config.USE_DOMAIN_ADAPTATION:
        plt.plot(history.history['emotion_accuracy'])
        plt.plot(history.history['val_emotion_accuracy'])
        if fine_tuning_history:
            plt.plot(np.arange(len(history.history['emotion_accuracy']), 
                               len(history.history['emotion_accuracy']) + len(fine_tuning_history.history['emotion_accuracy'])),
                     fine_tuning_history.history['emotion_accuracy'])
            plt.plot(np.arange(len(history.history['val_emotion_accuracy']), 
                               len(history.history['val_emotion_accuracy']) + len(fine_tuning_history.history['val_emotion_accuracy'])),
                     fine_tuning_history.history['val_emotion_accuracy'])
    else:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        if fine_tuning_history:
            plt.plot(np.arange(len(history.history['accuracy']), 
                               len(history.history['accuracy']) + len(fine_tuning_history.history['accuracy'])),
                     fine_tuning_history.history['accuracy'])
            plt.plot(np.arange(len(history.history['val_accuracy']), 
                               len(history.history['val_accuracy']) + len(fine_tuning_history.history['val_accuracy'])),
                     fine_tuning_history.history['val_accuracy'])
            
    plt.title('모델 정확도')
    plt.ylabel('정확도')
    plt.xlabel('Epoch')
    plt.legend(['훈련', '검증', '미세조정 훈련', '미세조정 검증'], loc='lower right')
    
    # 손실 그래프
    plt.subplot(1, 2, 2)
    if config.USE_DOMAIN_ADAPTATION:
        plt.plot(history.history['emotion_loss'])
        plt.plot(history.history['val_emotion_loss'])
        if fine_tuning_history:
            plt.plot(np.arange(len(history.history['emotion_loss']), 
                               len(history.history['emotion_loss']) + len(fine_tuning_history.history['emotion_loss'])),
                     fine_tuning_history.history['emotion_loss'])
            plt.plot(np.arange(len(history.history['val_emotion_loss']), 
                               len(history.history['val_emotion_loss']) + len(fine_tuning_history.history['val_emotion_loss'])),
                     fine_tuning_history.history['val_emotion_loss'])
    else:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        if fine_tuning_history:
            plt.plot(np.arange(len(history.history['loss']), 
                               len(history.history['loss']) + len(fine_tuning_history.history['loss'])),
                     fine_tuning_history.history['loss'])
            plt.plot(np.arange(len(history.history['val_loss']), 
                               len(history.history['val_loss']) + len(fine_tuning_history.history['val_loss'])),
                     fine_tuning_history.history['val_loss'])
    
    plt.title('모델 손실')
    plt.ylabel('손실')
    plt.xlabel('Epoch')
    plt.legend(['훈련', '검증', '미세조정 훈련', '미세조정 검증'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'training_history.png'))
    plt.close()