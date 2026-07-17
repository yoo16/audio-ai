#!/usr/bin/env python3
"""
tts.py - 学習済み GPT-SoVITS モデルで「テキスト -> 音声(wav)」を生成する CLI。

WebUI を使わずコマンドで自分の声を合成する。GPT-SoVITS の TTS_infer_pack を利用。

使い方 (GPT-SoVITS の venv で実行すること):
    cd ~/projects/GPT-SoVITS
    source .venv/bin/activate
    python /home/yoo/projects/audio-ai/tts.py "こんにちは、私の声です" -o out.wav

    # パラメータ指定
    python tts.py "テキスト" -o out.wav --temperature 0.7 --top-k 8 --cut cut2

詳細: docs/03-inference.md
"""
import os
import sys
import argparse

# ---- GPT-SoVITS のパス ----
GSV_DIR = os.environ.get("GSV_DIR", os.path.expanduser("~/projects/GPT-SoVITS"))

# ---- torchcodec/CUDA 用に LD_LIBRARY_PATH を通してから自身を再起動 ----
# (動的ローダは起動時にしか LD_LIBRARY_PATH を読まないため、env を設定して execv)
if not os.environ.get("_TTS_LDPATH_SET"):
    nv = os.path.join(GSV_DIR, ".venv/lib/python3.12/site-packages/nvidia")
    if os.path.isdir(nv):
        libdirs = [os.path.join(nv, d, "lib") for d in os.listdir(nv)
                   if os.path.isdir(os.path.join(nv, d, "lib"))]
        os.environ["LD_LIBRARY_PATH"] = ":".join(libdirs + [os.environ.get("LD_LIBRARY_PATH", "")])
    os.environ["_TTS_LDPATH_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ---- 再起動後: GPT-SoVITS を import できるようにする ----
ORIG_CWD = os.getcwd()          # ユーザーが実行したディレクトリ (出力パスの基準)
os.chdir(GSV_DIR)
sys.path.insert(0, GSV_DIR)

import soundfile as sf
import numpy as np

# ===== デフォルト設定 =====
DEFAULTS = {
    "version": "v2Pro",
    "gpt_model": "GPT_weights_v2Pro/yoo-e15.ckpt",
    "sovits_model": "SoVITS_weights_v2Pro/yoo_e8_s200.pth",
    "ref_audio": "output/slicer_opt/login_1.wav_0000021760_0000177280.wav",
    "ref_text": "第12回第1項ログイン処理その1について講義します",
    "bert": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    "cnhubert": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    "base_t2s": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "base_vits": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
}


def build_parser():
    p = argparse.ArgumentParser(description="GPT-SoVITS TTS CLI (自分の声で読み上げ)")
    p.add_argument("text", help="合成するテキスト")
    p.add_argument("-o", "--output", default="output.wav", help="出力 wav パス (既定: output.wav)")
    p.add_argument("--lang", default="ja", help="テキスト言語 (既定: ja)")
    # モデル
    p.add_argument("--gpt-model", default=DEFAULTS["gpt_model"], help="GPT モデル(.ckpt)")
    p.add_argument("--sovits-model", default=DEFAULTS["sovits_model"], help="SoVITS モデル(.pth)")
    p.add_argument("--version", default=DEFAULTS["version"], help="モデルバージョン (既定: v2Pro)")
    # 参照音声
    p.add_argument("--ref-audio", default=DEFAULTS["ref_audio"], help="参照音声")
    p.add_argument("--ref-text", default=DEFAULTS["ref_text"], help="参照音声のテキスト")
    p.add_argument("--ref-lang", default="ja", help="参照テキストの言語 (既定: ja)")
    # 生成パラメータ (安定寄りの既定値)
    p.add_argument("--top-k", type=int, default=8, help="top_k (既定: 8)")
    p.add_argument("--top-p", type=float, default=0.8, help="top_p (既定: 0.8)")
    p.add_argument("--temperature", type=float, default=0.7, help="temperature (既定: 0.7)")
    p.add_argument("--repetition-penalty", type=float, default=1.35, help="繰り返しペナルティ")
    p.add_argument("--speed", type=float, default=1.0, help="話速 (既定: 1.0)")
    p.add_argument("--cut", default="cut2",
                   help="テキスト分割方法 cut0..cut5 (既定: cut2=句読点等で分割)")
    p.add_argument("--device", default="cuda", help="cuda | cpu")
    p.add_argument("--fp16", action="store_true", default=True, help="半精度を使う (既定: ON)")
    p.add_argument("--no-fp16", dest="fp16", action="store_false", help="半精度を使わない")
    return p


def main():
    args = build_parser().parse_args()

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    # 事前学習(ベース)モデルを含む設定。fine-tune 済みモデルは後で上書きロード。
    config_dict = {
        "custom": {
            "device": args.device,
            "is_half": bool(args.fp16) and args.device == "cuda",
            "version": args.version,
            "t2s_weights_path": DEFAULTS["base_t2s"],
            "vits_weights_path": DEFAULTS["base_vits"],
            "cnhuhbert_base_path": DEFAULTS["cnhubert"],
            "bert_base_path": DEFAULTS["bert"],
        }
    }

    print(f"[tts] モデル初期化 (version={args.version}, device={args.device}) ...")
    tts = TTS(TTS_Config(config_dict))

    # 学習済みモデルを上書きロード
    print(f"[tts] GPT   : {args.gpt_model}")
    tts.init_t2s_weights(args.gpt_model)
    print(f"[tts] SoVITS: {args.sovits_model}")
    tts.init_vits_weights(args.sovits_model)

    inputs = {
        "text": args.text,
        "text_lang": args.lang,
        "ref_audio_path": args.ref_audio,
        "prompt_text": args.ref_text,
        "prompt_lang": args.ref_lang,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "speed_factor": args.speed,
        "text_split_method": args.cut,
        "batch_size": 1,
    }

    print(f"[tts] 合成中: {args.text!r}")
    sr = None
    chunks = []
    for item in tts.run(inputs):
        # run() は (sampling_rate, np.ndarray) を yield する
        this_sr, audio = item
        sr = this_sr
        chunks.append(audio)

    if not chunks:
        print("[tts] ERROR: 音声が生成されませんでした", file=sys.stderr)
        sys.exit(1)

    audio = np.concatenate(chunks, axis=0)
    out = args.output if os.path.isabs(args.output) else os.path.join(ORIG_CWD, args.output)
    sf.write(out, audio, sr)
    print(f"[tts] 完了: {out}  ({len(audio)/sr:.1f}s, {sr}Hz)")


if __name__ == "__main__":
    main()
