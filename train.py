import warnings

warnings.filterwarnings('ignore')

from ultralytics import YOLO

if __name__ == '__main__':
    # 模型配置位于 ultralytics/cfg/models/11/yolo11n_psconv_cbam_fem_bifpn_aligned.yaml
    model = YOLO('yolo11n_psconv_cbam_fem_bifpn_aligned.yaml')
    model.train(data='sea.yaml',
                imgsz=640,  # 参考实验使用 1000，可按显存情况调整
                epochs=300,
                batch=16,
                close_mosaic=10,  # 最后 10 个 epoch 关闭 mosaic
                workers=4,  # Windows 下卡住可改为 0
                device='0',  # 单卡；多卡见 ultralytics 文档
                optimizer='Adam',
                # patience=0,  # 设为 0 关闭早停
                # resume=True,  # 断点续训时改为 True 并指定 last.pt
                # amp=False,  # loss 出现 nan 时可关闭
                cache=False,
                project='runs/sea',
                name='exp',
                )
