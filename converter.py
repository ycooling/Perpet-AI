"""
모델 변환 및 TensorFlow.js 배포 유틸리티
"""

import os
import subprocess
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import tensorflowjs as tfjs
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config

def optimize_model_for_conversion(model_path):
    """
    TensorFlow.js 변환을 위한 모델 최적화
    """
    print("모델 최적화 중...")
    
    # 모델 로드
    model = load_model(model_path)
    
    # 도메인 적응 모델인 경우, 감정 분류 부분만 추출
    if config.USE_DOMAIN_ADAPTATION:
        emotion_output = model.get_layer('emotion').output
        model = tf.keras.Model(inputs=model.input, outputs=emotion_output)
    
    # 추론 전용 설정으로 모델 최적화
    model_for_export = tf.keras.models.clone_model(model)
    model_for_export.set_weights(model.get_weights())
    
    # 최적화된 모델 저장
    optimized_model_path = os.path.join(config.MODEL_DIR, 'model_optimized.keras')
    model_for_export.save(optimized_model_path)
    
    print(f"최적화된 모델 저장 완료: {optimized_model_path}")
    
    return optimized_model_path

def convert_to_tfjs(model_path):
    """
    Keras 모델을 TensorFlow.js 형식으로 변환
    """
    print("TensorFlow.js 변환 중...")
    
    # 변환 디렉토리 생성
    os.makedirs(config.TFJS_MODEL_DIR, exist_ok=True)
    
    # 변환 명령 실행
    try:
        # 직접 Python API 사용
        tfjs.converters.save_keras_model(
            load_model(model_path),
            config.TFJS_MODEL_DIR,
            quantization_dtype=None if config.QUANTIZATION_BYTES == 4 else np.uint8 if config.QUANTIZATION_BYTES == 1 else np.uint16,
            weight_shard_size_bytes=config.WEIGHT_SHARD_SIZE_BYTES
        )
        print(f"모델 변환 완료. 저장 위치: {config.TFJS_MODEL_DIR}")
        return True
    except Exception as e:
        print(f"모델 변환 실패: {e}")
        
        # 대체 방법: 명령줄 도구 사용
        try:
            quant_option = f"--quantization_bytes {config.QUANTIZATION_BYTES}"
            shard_option = f"--weight_shard_size_bytes {config.WEIGHT_SHARD_SIZE_BYTES}"
            
            cmd = f"tensorflowjs_converter --input_format keras {quant_option} {shard_option} {model_path} {config.TFJS_MODEL_DIR}"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            
            print(f"명령줄 도구로 모델 변환 완료: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"명령줄 도구로도 변환 실패: {e.stderr}")
            return False

def verify_converted_model(validation_generator=None):
    """
    변환된 모델의 정확도 검증 (옵션)
    """
    if validation_generator is None:
        print("검증 데이터가 제공되지 않아 모델 검증을 건너뜁니다.")
        return None
    
    print("변환된 모델 검증 중...")
    print("참고: TensorFlow.js 모델의 정확한 검증은 웹 환경에서 수행해야 합니다.")
    print("이 함수는 참고용 안내만 제공합니다.")
    
    # 메타데이터 확인
    try:
        with open(os.path.join(config.TFJS_MODEL_DIR, 'model.json'), 'r') as f:
            model_json = json.load(f)
        
        print(f"모델 구조: {len(model_json['modelTopology']['model_config']['layers'])} 레이어")
        print(f"가중치 샤드: {len(model_json['weightsManifest'][0]['paths'])} 파일")
        
        # 웹에서 사용할 메타데이터 저장
        model_info = {
            "image_size": config.IMAGE_SIZE,
            "num_classes": config.NUM_CLASSES,
            "base_model": config.BASE_MODEL,
            "quantization_bytes": config.QUANTIZATION_BYTES
        }
        
        with open(os.path.join(config.TFJS_MODEL_DIR, 'model_info.json'), 'w') as f:
            json.dump(model_info, f)
        
        print(f"모델 정보 저장 완료: {os.path.join(config.TFJS_MODEL_DIR, 'model_info.json')}")
        
        return True
    except Exception as e:
        print(f"메타데이터 확인 중 오류 발생: {e}")
        return False

def create_web_demo():
    """
    웹 데모를 위한 HTML/JS 파일 생성 (기본 템플릿)
    """
    print("웹 데모 파일 생성 중...")
    
    # 클래스 정보 로드
    with open(os.path.join(config.OUTPUT_DIR, 'class_indices.json'), 'r') as f:
        class_indices = json.load(f)
    
    # 클래스 이름 가져오기
    class_names = list(class_indices.keys())
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>반려견 감정 인식 모델</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.15.0"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .prediction-container {{ margin-top: 20px; }}
        .emotion {{ margin-bottom: 10px; }}
        .emotion-bar {{ height: 20px; background-color: #4CAF50; }}
        .dropzone {{ border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }}
        #preview {{ max-width: 300px; max-height: 300px; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>반려견 감정 인식</h1>
        
        <div class="dropzone" id="dropzone">
            <p>이미지를 드래그하거나 클릭하여 선택하세요</p>
            <input type="file" id="file-input" accept="image/*" style="display: none;">
            <img id="preview" style="display: none;">
        </div>
        
        <div class="prediction-container" id="prediction-container" style="display: none;">
            <h2>감정 분석 결과</h2>
            <div id="predictions"></div>
        </div>
    </div>

    <script>
        let model;
        const classNames = {json.stringify(class_names)};
        
        // 모델 로드
        async function loadModel() {{
            try {{
                model = await tf.loadLayersModel('./tfjs_model/model.json');
                console.log('모델 로드 완료');
            }} catch (error) {{
                console.error('모델 로드 오류:', error);
            }}
        }}
        
        // 이미지 전처리 및 예측
        async function predict(imgElement) {{
            // 이미지를 텐서로 변환
            const image = tf.browser.fromPixels(imgElement)
                .resizeBilinear([{config.IMAGE_SIZE[0]}, {config.IMAGE_SIZE[1]}])
                .div(255.0)
                .expandDims();
            
            // 예측 수행
            const predictions = await model.predict(image).data();
            
            // 텐서 메모리 해제
            image.dispose();
            
            // 결과 표시
            displayResults(predictions);
        }}
        
        // 결과 표시
        function displayResults(predictions) {{
            const predictionContainer = document.getElementById('prediction-container');
            const predictionsElement = document.getElementById('predictions');
            
            // 결과 컨테이너 보이기
            predictionContainer.style.display = 'block';
            
            // 기존 내용 초기화
            predictionsElement.innerHTML = '';
            
            // 예측 결과 정렬
            const predictionArray = Array.from(predictions).map((prob, i) => {{
                return {{ probability: prob, className: classNames[i] }};
            }});
            
            // 확률 내림차순 정렬
            predictionArray.sort((a, b) => b.probability - a.probability);
            
            // 결과 표시
            predictionArray.forEach(prediction => {{
                const percent = Math.round(prediction.probability * 100);
                
                const emotionDiv = document.createElement('div');
                emotionDiv.className = 'emotion';
                
                emotionDiv.innerHTML = `
                    <div>${{prediction.className}}: ${{percent}}%</div>
                    <div class="emotion-bar" style="width: ${{percent}}%;"></div>
                `;
                
                predictionsElement.appendChild(emotionDiv);
            }});
        }}
        
        // 파일 입력 처리
        function handleFileSelect(evt) {{
            const file = evt.target.files[0];
            
            if (file) {{
                const reader = new FileReader();
                
                reader.onload = function(e) {{
                    const preview = document.getElementById('preview');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    
                    // 이미지 로드 완료 후 예측
                    preview.onload = function() {{
                        predict(preview);
                    }};
                }};
                
                reader.readAsDataURL(file);
            }}
        }}
        
        // 드래그 앤 드롭 이벤트 처리
        function handleDragOver(evt) {{
            evt.stopPropagation();
            evt.preventDefault();
            evt.dataTransfer.dropEffect = 'copy';
        }}
        
        function handleDrop(evt) {{
            evt.stopPropagation();
            evt.preventDefault();
            
            const files = evt.dataTransfer.files;
            
            if (files.length > 0) {{
                document.getElementById('file-input').files = files;
                handleFileSelect({{ target: {{ files: files }} }});
            }}
        }}
        
        // 페이지 로드 시 초기화
        window.onload = function() {{
            // 모델 로드
            loadModel();
            
            // 파일 입력 이벤트 리스너
            document.getElementById('file-input').addEventListener('change', handleFileSelect, false);
            
            // 드롭존 클릭 시 파일 선택 다이얼로그 표시
            document.getElementById('dropzone').addEventListener('click', function() {{
                document.getElementById('file-input').click();
            }});
            
            // 드래그 앤 드롭 이벤트 리스너
            const dropzone = document.getElementById('dropzone');
            dropzone.addEventListener('dragover', handleDragOver, false);
            dropzone.addEventListener('drop', handleDrop, false);
        }};
    </script>
</body>
</html>
"""
    
    # HTML 파일 저장
    with open(os.path.join(config.TFJS_MODEL_DIR, 'index.html'), 'w') as f:
        f.write(html_content)
    
    print(f"웹 데모 파일 생성 완료: {os.path.join(config.TFJS_MODEL_DIR, 'index.html')}")
    
    return True

def export_model():
    """
    전체 모델 변환 및 내보내기 프로세스
    """
    # 모델 경로
    model_path = os.path.join(config.MODEL_DIR, 'model_final.keras')
    
    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return False
    
    # 1. 모델 최적화
    optimized_model_path = optimize_model_for_conversion(model_path)
    
    # 2. TensorFlow.js로 변환
    conversion_success = convert_to_tfjs(optimized_model_path)
    
    if not conversion_success:
        print("모델 변환에 실패했습니다.")
        return False
    
    # 3. 변환된 모델 검증
    verification_success = verify_converted_model()
    
    # 4. 웹 데모 생성
    create_web_demo()
    
    print("모델 내보내기가 완료되었습니다.")
    print(f"TensorFlow.js 모델 및 웹 데모가 저장된 위치: {config.TFJS_MODEL_DIR}")
    print("웹 서버에서 이 폴더를 호스팅하여 모델을 테스트할 수 있습니다.")
    
    return True