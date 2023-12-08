TASK=$1

PYTHONPATH=. python source/visualize.py \
    --data_dir "/home/mpham/workspace/huawei-time-series/data/InterviewCaseStudies/$TASK" \
    --out_dir "/home/mpham/workspace/huawei-time-series/results/$TASK" \
    --config_file "/home/mpham/workspace/huawei-time-series/configs/$TASK.yaml" \
    --task $TASK