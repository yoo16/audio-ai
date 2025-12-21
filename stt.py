import os
import glob
from faster_whisper import WhisperModel

# モデル初期化 (RTX 5080)
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

input_dir = "audio/"
output_file = "transcriptions.list" # 学習用ラベルファイル

# .wavファイル取得
wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))

print(f"処理開始: {len(wav_files)} 件")

with open(output_file, "w", encoding="utf-8") as f:
    for wav_path in wav_files:
        # 推論
        segments, _ = model.transcribe(wav_path, language="ja")
        text = "".join([s.text for s in segments])
        
        # GPT-SoVITS形式: パス|名前|言語|テキスト
        line = f"{wav_path}|yoo|JP|{text}"
        
        f.write(line + "\n")
        print(f"完了: {wav_path} -> {text}")

print(f"\n書き出し完了: {output_file}")