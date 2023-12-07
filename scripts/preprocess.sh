TASK=$1

PYTHONPATH=. python source/preprocess.py \
    --data_dir "/home/mpham/workspace/huawei-time-series/data/InterviewCaseStudies/$TASK" \
    --out_dir "/home/mpham/workspace/huawei-time-series/data/processed/$TASK" \
    --task $TASK