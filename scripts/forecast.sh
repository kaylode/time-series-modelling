ID=$1
PYTHONPATH=. python source/forecast.py \
    --id ${ID:=-1} \
    --data_dir "data/processed/AD&P" \
    --out_dir "results/AD&P" \
    --config_file "configs/AD&P.yaml"
