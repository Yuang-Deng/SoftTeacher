_base_ = "base.py"

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="/data/dya/dataset/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="/data/dya/dataset/coco/train2017/",
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="/data/dya/dataset/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="/data/dya/dataset/coco/train2017/",
        ),
    ),
    val=dict(
        ann_file="/data/dya/dataset/coco/annotations/instances_val2017.json",
        img_prefix="/data/dya/dataset/coco/val2017/",
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
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
        # dict(
        #     type="WandbLoggerHook",
        #     init_kwargs=dict(
        #         project="pre_release",
        #         name="${cfg_name}",
        #         config=dict(
        #             fold="${fold}",
        #             percent="${percent}",
        #             work_dirs="${work_dir}",
        #             total_step="${runner.max_iters}",
        #         ),
        #     ),
        #     by_epoch=False,
        # ),
    ],
)
