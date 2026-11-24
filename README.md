# SDE + CC-TDH

## Paper Overview

This repository implements SDE + CC-TDH, a two-stage expert knowledge transfer
method for audio-visual question answering. It supports the Music-AVQA and AVQA
data protocols and provides runtimes for Qwen2.5-Omni and
MiniCPM-o-4.5-GPTQ.

The complete workflow consists of the following stages:

1. Build visual, audio, and temporal experts with ImageBind, MuQ, and VideoMAE.
2. Construct per-example RPU states and repair pools from Base and expert predictions.
3. Train the first LoRA adapter during SDE to repair target examples while preserving retained responses.
4. Generate detail supervision through local intervention validation, routing, and teacher fitting.
5. Freeze the SDE adapter and train a residual LoRA adapter during CC-TDH.
6. Load both adapters for inference, metric computation, and deployment export.

Training uses source-disjoint data partitions. SDE, CC-TDH, routing, retention
basis construction, and evaluation consistently use seed 18. Inference uses the
Base audio-visual model and trained adapters without loading expert encoders or
the teacher bundle.

## Environment Setup

### System Requirements

The environments require an NVIDIA GPU with a compatible driver, Conda, Git,
FFmpeg, and FFprobe. Install the project after entering the repository:

```bash
cd /path/to/sde_cctdh
python -m pip install -e .
```

### Qwen2.5-Omni Environment

```bash
bash environments/create.sh qwen
conda activate sde-cctdh-qwen
python scripts/check_environment.py qwen
```

### MiniCPM-o-4.5-GPTQ Environment

The MiniCPM environment uses Python 3.11, PyTorch 2.9.1+cu128,
Transformers 4.51.0, and GPTQModel 2.2.0:

```bash
conda env create -f environments/minicpm.yaml
conda activate sde-cctdh-minicpm
python -m pip install -e .
```

### Expert Environments

ImageBind and MuQ/VideoMAE use separate environments:

```bash
bash environments/create.sh imagebind
bash environments/create.sh temporal_experts

conda run -n sde-cctdh-imagebind \
  python scripts/check_environment.py imagebind
conda run -n sde-cctdh-temporal \
  python scripts/check_environment.py temporal
```

## Dataset Preparation

### Download the Datasets

```bash
python dataset/download_datasets.py music_avqa
python dataset/download_datasets.py avqa
```

Place Music-AVQA videos in `dataset/Music_AVQA/media` and AVQA media in
`dataset/AVQA/media`. The preparation script generates source-disjoint Adapt,
Tune, and Holdout manifests.

Music-AVQA example:

```bash
python dataset/prepare_dataset.py \
  --dataset music_avqa \
  --input adapt=dataset/Music_AVQA/hf_annotations/adapt.json \
  --input tune=dataset/Music_AVQA/hf_annotations/tune.json \
  --input holdout=dataset/Music_AVQA/hf_annotations/holdout.json \
  --media-root dataset/Music_AVQA/media \
  --protocol-root dataset/Music_AVQA/protocol
```

Use the same command structure for AVQA, replacing the dataset name, annotation
inputs, media directory, and protocol output directory with the corresponding
AVQA paths.

## Model Preparation

Download the audio-visual language models:

```bash
python model/download_models.py avllm qwen_omni_3b
python model/download_models.py avllm qwen_omni_7b
python model/download_models.py avllm minicpm_o_45_gptq
```

Download the expert models and external baseline weights:

```bash
python model/download_models.py expert all
python model/download_models.py external qa_tiger
```

Models are stored under `model/avllm`, `model/experts`, and `model/baselines`.
Model sources and local paths are defined in `model/config.yaml`.

## Expert Features and Heads

The following examples use Music-AVQA:

```bash
MODEL_ROOT=model
DATA=dataset/Music_AVQA/protocol
```

### ImageBind Features

```bash
conda run -n sde-cctdh-imagebind \
  python expert/extract_raw_features.py text \
  --dataset music_avqa \
  --protocol-root "$DATA" \
  --imagebind-source "$MODEL_ROOT/experts/ImageBind/code" \
  --imagebind-checkpoint "$MODEL_ROOT/experts/ImageBind/imagebind_huge.pth"

conda run -n sde-cctdh-imagebind \
  python expert/extract_raw_features.py imagebind \
  --dataset music_avqa \
  --protocol-root "$DATA" \
  --imagebind-source "$MODEL_ROOT/experts/ImageBind/code" \
  --imagebind-checkpoint "$MODEL_ROOT/experts/ImageBind/imagebind_huge.pth"
```

### MuQ and VideoMAE Features

```bash
conda run -n sde-cctdh-temporal \
  python expert/extract_raw_features.py muq-videomae \
  --dataset music_avqa \
  --protocol-root "$DATA" \
  --muq-model "$MODEL_ROOT/experts/MuQ/MuQ-large-msd-iter" \
  --videomae-model "$MODEL_ROOT/experts/VideoMAE/hf_videomae_base"

conda run -n sde-cctdh-temporal \
  python expert/extract_raw_features.py pack-music \
  --protocol-root "$DATA"
```

### Train Expert Heads

```bash
conda run -n sde-cctdh-temporal \
  python expert/music_avqa_42.py \
  --expert imagebind \
  --arrays "$DATA/features/music_arrays/imagebind_arrays.npz" \
  --manifest "$DATA/features/music_arrays/music_manifest.jsonl" \
  --output runs/experts/imagebind
```

Replace `imagebind` with `muq` and `videomae` to train the other two expert
heads. For AVQA, use `expert/avqa_4way.py` to train the shared four-candidate
scorer.

## Base Decoding and RPU Construction

Select a model runtime before starting the main pipeline. For Qwen:

```bash
conda activate sde-cctdh-qwen
MODEL=model/avllm/Qwen2.5-Omni-3B
CONFIG=configs/qwen3b.yaml
PIPELINE=scripts/qwen_pipeline.py
RUNTIME=sde_cctdh.qwen:build_runtime
RUN=runs/qwen_music
mkdir -p "$RUN"
```

For MiniCPM:

```bash
conda activate sde-cctdh-minicpm
MODEL=model/avllm/MiniCPM-o-4_5-GPTQ
CONFIG=configs/minicpm.yaml
PIPELINE=scripts/minicpm_pipeline.py
RUNTIME=sde_cctdh.minicpm:create_runtime
RUN=runs/minicpm_music
mkdir -p "$RUN"
```

Generate Base responses and the response cache:

```bash
python "$PIPELINE" decode \
  --config "$CONFIG" \
  --model-path "$MODEL" \
  --manifest "$DATA/manifests/adapt/inference.jsonl" \
  --labels "$DATA/manifests/adapt/labels.jsonl" \
  --output "$RUN/base_outputs.jsonl"
```

Combine out-of-fold predictions from the three experts:

```bash
python "$PIPELINE" expert-pairs \
  --base-outputs "$RUN/base_outputs.jsonl" \
  --metadata "$DATA/data/full_qa_manifest.json" \
  --expert-predictions runs/experts/imagebind/imagebind_train_oof.jsonl \
  --expert-predictions runs/experts/muq/muq_train_oof.jsonl \
  --expert-predictions runs/experts/videomae/videomae_train_oof.jsonl \
  --output "$RUN/expert_pairs.jsonl"
```

Build per-example RPU states, the pair audit, and SDE inputs:

```bash
python scripts/build_rpu_manifest.py \
  --expert-pairs "$RUN/expert_pairs.jsonl" \
  --pair-output "$RUN/pair_audit.jsonl" \
  --example-output "$RUN/example_rpu.jsonl"
```

## SDE Training

The common training entry point loads either the Qwen or MiniCPM runtime through
`--runtime`:

```bash
python scripts/train_sde.py \
  --config "$CONFIG" \
  --examples "$RUN/example_rpu.jsonl" \
  --responses "$RUN/base_response_cache.jsonl" \
  --runtime "$RUNTIME" \
  --runtime-arg model_path="$MODEL" \
  --runtime-arg protocol_config=baseline/config.yaml \
  --batch-size 1 \
  --basis-batch-size 1 \
  --gradient-clip-norm 1.0 \
  --seed 18 \
  --require-resolved \
  --output "$RUN/sde"
```

The output contains the SDE adapter, three training pools, the training log, and
the retention basis. Load the SDE adapter to generate frozen-parent responses:

```bash
python "$PIPELINE" decode \
  --config "$CONFIG" \
  --model-path "$MODEL" \
  --manifest "$DATA/manifests/adapt/inference.jsonl" \
  --labels "$DATA/manifests/adapt/labels.jsonl" \
  --sde-adapter "$RUN/sde/adapter" \
  --output "$RUN/parent_outputs.jsonl"
```

## Local Intervention Validation

Run local validation for each expert. ImageBind example:

```bash
conda run -n sde-cctdh-imagebind \
  python expert/local_validation.py validate \
  --expert imagebind \
  --protocol-root "$DATA" \
  --head-dir runs/experts/imagebind \
  --pair-audit "$RUN/pair_audit.jsonl" \
  --imagebind-source "$MODEL_ROOT/experts/ImageBind/code" \
  --imagebind-checkpoint "$MODEL_ROOT/experts/ImageBind/imagebind_huge.pth" \
  --cache "$RUN/local_cache" \
  --output "$RUN/local_validation"
```

Run MuQ and VideoMAE validation in the temporal environment and select the
corresponding model with `--model-path`. Merge the three expert results after
validation:

```bash
python expert/local_validation.py merge \
  --input "$RUN/local_validation" \
  --output "$RUN/local_validation"
```

## CC-TDH Data and Teacher Fitting

Generate the residual-repair, parent-protect, parent-anchor, teacher-fit, and
detail-supervision artifacts:

```bash
python "$PIPELINE" prepare-cctdh \
  --config "$CONFIG" \
  --sde-repair-pool "$RUN/sde/pools/repair.jsonl" \
  --examples "$RUN/example_rpu.jsonl" \
  --pair-audit "$RUN/pair_audit.jsonl" \
  --parent-outputs "$RUN/parent_outputs.jsonl" \
  --intervention-effects "$RUN/local_validation/intervention_effects.jsonl" \
  --local-details "$RUN/local_validation/local_details.jsonl" \
  --output "$RUN/cctdh_data"
```

Qwen uses hidden layer 30, while MiniCPM uses hidden layer 34. Select the value
from the active model configuration:

```bash
if [ "$CONFIG" = "configs/minicpm.yaml" ]; then
  HIDDEN_LAYER=34
else
  HIDDEN_LAYER=30
fi
```

Extract frozen-parent hidden states at the answer position:

```bash
python "$PIPELINE" extract-parent-hidden \
  --config "$CONFIG" \
  --model-path "$MODEL" \
  --manifest "$RUN/cctdh_data/teacher_fit.jsonl" \
  --sde-adapter "$RUN/sde/adapter" \
  --hidden-layer "$HIDDEN_LAYER" \
  --output "$RUN/teacher_fit_hidden.jsonl"
```

Fit the source-disjoint teacher bundle:

```bash
python "$PIPELINE" fit-teacher \
  --teacher-fit "$RUN/cctdh_data/teacher_fit.jsonl" \
  --parent-hidden "$RUN/teacher_fit_hidden.jsonl" \
  --supervision "$RUN/cctdh_data/validated_repair.jsonl" \
  --hidden-layer "$HIDDEN_LAYER" \
  --ridge 1.0 \
  --output "$RUN/teacher_bundle.pt"
```

## CC-TDH Residual Adapter Training

This stage loads and freezes the SDE adapter, then trains only the residual
LoRA adapter:

```bash
python scripts/train_cctdh.py \
  --config "$CONFIG" \
  --validated-repair "$RUN/cctdh_data/validated_repair.jsonl" \
  --protect "$RUN/cctdh_data/parent_protect.jsonl" \
  --anchor "$RUN/cctdh_data/parent_anchor.jsonl" \
  --runtime "$RUNTIME" \
  --runtime-arg model_path="$MODEL" \
  --runtime-arg protocol_config=baseline/config.yaml \
  --runtime-arg sde_adapter="$RUN/sde/adapter" \
  --runtime-arg teacher_bundle="$RUN/teacher_bundle.pt" \
  --batch-size 1 \
  --basis-batch-size 1 \
  --gradient-clip-norm 1.0 \
  --seed 18 \
  --require-resolved \
  --output "$RUN/cctdh"
```

The output directory contains the residual adapter, parent-retention basis,
training pools, per-step log, and training summary.

## Final Inference

Load the SDE and CC-TDH adapters in order and decode the Holdout manifest:

```bash
python "$PIPELINE" decode \
  --config "$CONFIG" \
  --model-path "$MODEL" \
  --manifest "$DATA/manifests/holdout/inference.jsonl" \
  --labels "$DATA/manifests/holdout/labels.jsonl" \
  --sde-adapter "$RUN/sde/adapter" \
  --cctdh-adapter "$RUN/cctdh/adapter" \
  --output "$RUN/holdout_method_outputs.jsonl"
```

The `decode` command also supports Base-only decoding and SDE-parent decoding.

## Evaluation

After preparing the Holdout Base output, method output, and fixed RPU manifest,
run:

```bash
python "$PIPELINE" evaluate \
  --base-outputs "$RUN/holdout_base_outputs.jsonl" \
  --method-outputs "$RUN/holdout_method_outputs.jsonl" \
  --examples "$RUN/holdout_example_rpu.jsonl" \
  --seed 18 \
  --output "$RUN/holdout_metrics.json"
```

The evaluation output includes Accuracy, BER, DR, RR, PR, UFR, and
source-bootstrap statistics.

## Deployment Export

The Qwen pipeline merges both LoRA adapters and exports a standalone model
directory:

```bash
python scripts/qwen_pipeline.py export-deployed \
  --config configs/qwen3b.yaml \
  --model-path "$MODEL" \
  --sde-adapter "$RUN/sde/adapter" \
  --cctdh-adapter "$RUN/cctdh/adapter" \
  --output "$RUN/deployed_model"
```

The MiniCPM-GPTQ pipeline exports two lightweight adapters and a deployment
manifest that records their loading order:

```bash
python scripts/minicpm_pipeline.py export-deployed \
  --config configs/minicpm.yaml \
  --model-path "$MODEL" \
  --sde-adapter "$RUN/sde/adapter" \
  --cctdh-adapter "$RUN/cctdh/adapter" \
  --output "$RUN/deployed_adapters"
```

The MiniCPM export directory contains `sde_adapter.pt`, `cctdh_adapter.pt`, and
`deployment_manifest.json`. Load the adapters according to the manifest's
`load_order` field.

## Baselines

### Local Audio-Visual Model Inference

```bash
python baseline/infer_local.py \
  --model qwen_omni_7b \
  --manifest dataset/AVQA/protocol/manifests/holdout/inference.jsonl \
  --output runs/qwen7b_avqa.jsonl

python baseline/infer_local.py \
  --model minicpm_o_45_gptq \
  --manifest dataset/AVQA/protocol/manifests/holdout/inference.jsonl \
  --output runs/minicpm_avqa.jsonl
```

### Knowledge Distillation Baselines

Use `baseline/distillation.py prepare` to combine the manifest, expert
probabilities, expert features, and training exposure. Then train Vanilla KD,
RWKD, or UFA with low-rank expert projectors:

```bash
KD_MODEL=model/avllm/Qwen2.5-Omni-3B

python baseline/distillation.py train \
  --method vanilla_kd \
  --bundle runs/distillation_bundle \
  --model-path "$KD_MODEL" \
  --output runs/baselines/vanilla_kd
```

Replace `--method` with `rwkd` or `ufa` to train the other knowledge-transfer
baselines.

### API Model Inference

```bash
MINIMAX_API_KEY=... python baseline/infer_minimax.py \
  --manifest dataset/AVQA/protocol/manifests/holdout/inference.jsonl \
  --output runs/minimax_avqa.jsonl
```

## Configuration and Runtime Checks

Validate the model configurations before training:

```bash
python scripts/validate_config.py configs/qwen3b.yaml --require-resolved
python scripts/validate_config.py configs/minicpm.yaml --require-resolved
```

Run the smoke test and complete test suite:

```bash
python scripts/smoke_test.py
pytest -q
```

The main configuration files are:

- `configs/qwen3b.yaml`: Qwen SDE + CC-TDH configuration.
- `configs/minicpm.yaml`: MiniCPM-GPTQ SDE + CC-TDH configuration.
- `baseline/config.yaml`: data protocols, prompts, model paths, and generation settings.
- `dataset/config.yaml`: dataset download and preparation settings.
- `model/config.yaml`: model sources and local storage paths.
