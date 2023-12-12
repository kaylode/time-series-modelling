TASK=$1

PYTHONPATH=. python source/visualize.py \
    --data_dir "data/processed/$TASK" \
    --out_dir "results/$TASK" \
    --config_file "configs/$TASK.yaml" \
    --task $TASK