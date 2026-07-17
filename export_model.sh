#!/usr/bin/env bash
#
# GPT-SoVITS 学習済みモデルを TorchScript (.pt) にエクスポートするスクリプト。
#   - LD_LIBRARY_PATH (torchcodec/CUDA) を自動設定
#   - v2Pro 対応 (--version)
#   - 参照テキストを一時ファイル化して --ref_text に渡す
#
# 使い方:
#   ./export_model.sh
#   # または環境変数で上書き:
#   REF_TEXT="任意の参照テキスト" ./export_model.sh
#
# 詳細仕様: docs/06-export.md
set -euo pipefail

# ===== 設定 (必要に応じて書き換え) =====
GSV_DIR="${GSV_DIR:-$HOME/projects/GPT-SoVITS}"          # GPT-SoVITS 本体
VERSION="${VERSION:-v2Pro}"                               # モデルバージョン
GPT_MODEL="${GPT_MODEL:-GPT_weights_v2Pro/yoo-e15.ckpt}"  # GSV_DIR からの相対 or 絶対
SOVITS_MODEL="${SOVITS_MODEL:-SoVITS_weights_v2Pro/yoo_e8_s200.pth}"
REF_AUDIO="${REF_AUDIO:-output/slicer_opt/login_1.wav_0000021760_0000177280.wav}"
REF_TEXT="${REF_TEXT:-第12回第1項ログイン処理その1について講義します}"
OUTPUT_PATH="${OUTPUT_PATH:-exported/yoo}"                # 出力ディレクトリ
DEVICE="${DEVICE:-cuda}"                                  # cuda | cpu
EXPORT_COMMON="${EXPORT_COMMON:-1}"                       # 1=BERT/SSLも同梱, 0=しない
# =======================================

cd "$GSV_DIR"

# venv
if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "ERROR: $GSV_DIR/.venv が見つかりません" >&2
    exit 1
fi

# LD_LIBRARY_PATH (torchcodec が nvidia 各ライブラリを実行時に要求する)
NV="$PWD/.venv/lib/python3.12/site-packages/nvidia"
if [[ -d "$NV" ]]; then
    export LD_LIBRARY_PATH="$(ls -d "$NV"/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
fi

# 入力チェック
for f in "$GPT_MODEL" "$SOVITS_MODEL" "$REF_AUDIO"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: ファイルが見つかりません: $f" >&2
        exit 1
    fi
done

# 参照テキストを一時ファイル化 (--ref_text はファイルパスを要求する)
REF_TEXT_FILE="$(mktemp --suffix=.txt)"
trap 'rm -f "$REF_TEXT_FILE"' EXIT
printf '%s\n' "$REF_TEXT" > "$REF_TEXT_FILE"

# --export_common_model フラグ
COMMON_FLAG=()
if [[ "$EXPORT_COMMON" == "1" ]]; then
    COMMON_FLAG=(--export_common_model)
fi

echo "=== エクスポート開始 ==="
echo "  GPT   : $GPT_MODEL"
echo "  SoVITS: $SOVITS_MODEL"
echo "  参照  : $REF_AUDIO"
echo "  出力  : $OUTPUT_PATH  (version=$VERSION, device=$DEVICE)"

python GPT_SoVITS/export_torch_script.py \
    --gpt_model    "$GPT_MODEL" \
    --sovits_model "$SOVITS_MODEL" \
    --ref_audio    "$REF_AUDIO" \
    --ref_text     "$REF_TEXT_FILE" \
    --output_path  "$OUTPUT_PATH" \
    --version      "$VERSION" \
    --device       "$DEVICE" \
    "${COMMON_FLAG[@]}"

echo "=== 完了: $GSV_DIR/$OUTPUT_PATH ==="
ls -la "$OUTPUT_PATH"
