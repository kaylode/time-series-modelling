TASK=$1

PYTHONPATH=. python source/preprocess.py \
    --data_dir "data/InterviewCaseStudies/$TASK" \
    --out_dir "data/processed/$TASK" \
    --config_file "configs/$TASK.yaml" \
    --task $TASK