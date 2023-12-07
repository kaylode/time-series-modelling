TASK=$1

python preprocess.py \
    --data_dir "/home/mpham/workspace/huawei-time-series/data/InterviewCaseStudies/$TASK" \
    --out_dir "/home/mpham/workspace/huawei-time-series/data/preprocessed/$TASK" \
    --task $TASK