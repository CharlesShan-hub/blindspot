
uv run ../scripts/scene_median.py \
  --indir /Users/charles/workspace/data/blindlama/dataset/test/img \
  --maskdir /Users/charles/workspace/data/blindlama/dataset/test/mask \
  --outdir "$(pwd)/output_median" \
  --kernel-size 3