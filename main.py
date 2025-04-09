"""
반려견 감정 인식 모델 - 메인 실행 파일
"""

import os
import argparse
import tensorflow as tf
import preprocessing
import model
import training
import evaluation
import converter
import config
import json

def check_gpu():
    """
    GPU 가용성 확인
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"사용 가능한 GPU: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
        
        # GPU 메모리 증가 허용
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU 메모리 증가 허용 설정 완료")
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")
    else:
        print("사용 가능한 GPU가 없습니다. CPU로 실행됩니다.")

def train_pipeline():
    """
    모델 훈련 파이프라인 실행
    """
    print("\n===== 모델 훈련 파이프라인 시작 =====")
    
    # GPU 확인
    check_gpu()
    
    # 데이터 확인
    class_names = preprocessing.get_class_names()
    print(f"감정 클래스: {class_names}")
    print(f"총 클래스 수: {len(class_names)}")
    
    # 모델 훈련
    trained_model = training.train_model()
    
    print("===== 모델 훈련 파이프라인 완료 =====\n")
    
    return trained_model

def evaluate_pipeline(trained_model=None):
    """
    모델 평가 파이프라인 실행
    """
    print("\n===== 모델 평가 파이프라인 시작 =====")
    
    # 모델이 제공되지 않은 경우 저장된 모델 로드
    if trained_model is None:
        model_path = os.path.join(config.MODEL_DIR, 'model_final.keras')
        if os.path.exists(model_path):
            trained_model = tf.keras.models.load_model(model_path)
            print(f"저장된 모델 로드 완료: {model_path}")
        else:
            print(f"저장된 모델을 찾을 수 없습니다: {model_path}")
            return False
    
    # 검증 데이터 생성
    _, validation_generator = preprocessing.create_data_generators()
    
    # 모델 평가
    results = evaluation.evaluate_model(trained_model, validation_generator)
    
    # 예측 및 성능 보고서 생성
    pred_indices, true_indices = evaluation.get_predictions(trained_model, validation_generator)
    report = evaluation.generate_classification_report(pred_indices, true_indices)
    
    # 오분류 사례 분석
    evaluation.analyze_misclassifications(trained_model, validation_generator)
    
    print("===== 모델 평가 파이프라인 완료 =====\n")
    
    return True

def convert_pipeline():
    """
    모델 변환 파이프라인 실행
    """
    print("\n===== 모델 변환 파이프라인 시작 =====")
    
    # 모델 변환 및 내보내기
    success = converter.export_model()
    
    if success:
        print("===== 모델 변환 파이프라인 완료 =====\n")
    else:
        print("===== 모델 변환 파이프라인 실패 =====\n")
    
    return success

def setup_config(args):
    """
    커맨드 라인 인자로 설정 업데이트
    """
    if args.data_dir:
        config.DATA_DIR = args.data_dir
        print(f"데이터 디렉토리 설정: {config.DATA_DIR}")
    
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
        config.MODEL_DIR = os.path.join(config.OUTPUT_DIR, "models")
        config.LOG_DIR = os.path.join(config.OUTPUT_DIR, "logs")
        config.TFJS_MODEL_DIR = os.path.join(config.OUTPUT_DIR, "tfjs_model")
        
        # 필요한 디렉토리 생성
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        print(f"출력 디렉토리 설정: {config.OUTPUT_DIR}")
    
    if args.base_model:
        config.BASE_MODEL = args.base_model
        print(f"기반 모델 설정: {config.BASE_MODEL}")
    
    if args.attention:
        config.USE_ATTENTION = True
        print(f"주의 메커니즘 활성화: {config.USE_ATTENTION}")
    
    if args.domain_adaptation:
        config.USE_DOMAIN_ADAPTATION = True
        print(f"도메인 적응 활성화: {config.USE_DOMAIN_ADAPTATION}")
    
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
        print(f"배치 크기 설정: {config.BATCH_SIZE}")
    
    if args.epochs:
        config.EPOCHS = args.epochs
        print(f"에포크 수 설정: {config.EPOCHS}")
    
    if args.fine_tuning_epochs:
        config.FINE_TUNING_EPOCHS = args.fine_tuning_epochs
        print(f"미세조정 에포크 수 설정: {config.FINE_TUNING_EPOCHS}")
    
    if args.quantization_bytes:
        config.QUANTIZATION_BYTES = args.quantization_bytes
        print(f"양자화 바이트 설정: {config.QUANTIZATION_BYTES}")
    
    # 디렉토리 존재 확인
    if not os.path.exists(config.DATA_DIR):
        print(f"경고: 데이터 디렉토리를 찾을 수 없습니다: {config.DATA_DIR}")
        return False
    
    return True

def main():
    """
    메인 실행 함수
    """
    # 커맨드 라인 인자 파서
    parser = argparse.ArgumentParser(description='반려견 감정 인식 모델 훈련 및 배포')
    
    parser.add_argument('--data_dir', type=str, help='데이터셋 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, help='출력 디렉토리 경로')
    parser.add_argument('--base_model', type=str, choices=['EfficientNetV2S', 'MobileNetV2'], help='기반 모델 선택')
    parser.add_argument('--attention', action='store_true', help='주의 메커니즘 사용')
    parser.add_argument('--domain_adaptation', action='store_true', help='도메인 적응 훈련 사용')
    parser.add_argument('--batch_size', type=int, help='배치 크기')
    parser.add_argument('--epochs', type=int, help='훈련 에포크 수')
    parser.add_argument('--fine_tuning_epochs', type=int, help='미세조정 에포크 수')
    parser.add_argument('--quantization_bytes', type=int, choices=[1, 2, 4], help='TensorFlow.js 변환 양자화 바이트')
    
    # 실행 모드 선택
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'convert', 'all'], default='all',
                      help='실행 모드 (train: 훈련만, evaluate: 평가만, convert: 변환만, all: 전체 파이프라인)')
    
    # 인자 파싱
    args = parser.parse_args()
    
    # 설정 업데이트
    if not setup_config(args):
        return
    
    # 모드에 따른 실행
    if args.mode == 'train' or args.mode == 'all':
        trained_model = train_pipeline()
    else:
        trained_model = None
    
    if args.mode == 'evaluate' or args.mode == 'all':
        evaluate_pipeline(trained_model)
    
    if args.mode == 'convert' or args.mode == 'all':
        convert_pipeline()
    
    print("프로그램 실행 완료")

if __name__ == "__main__":
    main()