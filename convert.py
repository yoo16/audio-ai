import os
import glob
import soundfile as sf
import librosa

# GPT-SoVITS 用フォーマット
TARGET_SR = 32000      # サンプリングレート
input_dir = "audio/"
output_dir = "audio_converted/"

os.makedirs(output_dir, exist_ok=True)

wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
print(f"変換開始: {len(wav_files)} 件 -> {output_dir}")

total = 0.0
for wav_path in wav_files:
    # 読み込み (float32 でもOK)。mono=True でモノラル化
    audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    total += len(audio) / TARGET_SR

    out_path = os.path.join(output_dir, os.path.basename(wav_path))
    # 16bit PCM で書き出し
    sf.write(out_path, audio, TARGET_SR, subtype="PCM_16")
    print(f"完了: {out_path}  ({len(audio)/TARGET_SR:.1f}s)")

print(f"\n変換完了: {len(wav_files)}件 / 合計 {total:.1f}秒 ({total/60:.1f}分)")
