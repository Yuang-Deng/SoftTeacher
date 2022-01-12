_base_ = "base.py"

data_root = 'C:/Users/Alex/WorkSpace/dataset/VOCdevkit/'
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=0,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="/data/dya/workspace/cache/instances_train2017.1@10.json",
            img_prefix="/data/dya/dataset/coco/train2017/",
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="/data/dya/workspace/cache/instances_train2017.1@10-unlabeled.json",
            img_prefix="/data/dya/dataset/coco/train2017/",
        ),
    ),
    val=dict(
        ann_file="/data/dya/dataset/coco/annotations/instances_val2017.json",
        img_prefix="/data/dya/dataset/coco/val2017/",
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

work_dir = "work_dirs/${cfg_name}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
