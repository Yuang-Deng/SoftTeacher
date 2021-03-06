_base_ = "base.py"

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=0,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="C:/Users/Alex/WorkSpace/dataset/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="C:/Users/Alex/WorkSpace/dataset/coco/train2017/",
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="C:/Users/Alex/WorkSpace/dataset/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="C:/Users/Alex/WorkSpace/dataset/coco/train2017/",
        ),
    ),
    val=dict(
        type="CocoDataset",
        ann_file="C:/Users/Alex/WorkSpace/dataset/coco/annotations/instances_val2017.json",
        img_prefix="C:/Users/Alex/WorkSpace/dataset/coco/val2017/"
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="pre_release",
                name="${cfg_name}",
                config=dict(
                    fold="${fold}",
                    percent="${percent}",
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
