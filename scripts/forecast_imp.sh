ID=$1
PYTHONPATH=. python source/forecast.py \
    --id ${ID:=-1} \
    --data_dir "/home/mpham/workspace/huawei-time-series/results/AD&P/anomalies/imputed" \
    --out_dir "/home/mpham/workspace/huawei-time-series/results/AD&P" \
    --config_file "/home/mpham/workspace/huawei-time-series/configs/AD&P.yaml"
