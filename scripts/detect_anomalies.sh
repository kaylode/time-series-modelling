ID=$1

PYTHONPATH=. python source/anomalies.py \
    --id ${ID:=-1} \
    --data_dir "/home/mpham/workspace/huawei-time-series/data/processed/AD&P" \
    --out_dir "/home/mpham/workspace/huawei-time-series/results/AD&P/anomalies" \
    --config_file "/home/mpham/workspace/huawei-time-series/configs/AD&P.yaml"