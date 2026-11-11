import os
import json
from pathlib import Path
def select_centered_yolo_samples(dataset_path, num_samples=5, center_ratio=0.3):
    """
    从YOLO数据集中选择每个类别的中心样本

    Args:
        dataset_path: 数据集路径
        num_samples: 每个类别的样本数量
        center_ratio: 中心区域比例
    """
    dataset_path = Path(dataset_path)

    # 检查数据集路径是否存在
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return {}

    print(f"数据集路径: {dataset_path}")

    # 使用训练集的图像和标注
    images_dir = dataset_path / "train" / "images"
    labels_dir = dataset_path / "train" / "labels"

    # 检查目录是否存在
    if not images_dir.exists():
        print(f"错误: 图像目录不存在: {images_dir}")
        # 尝试其他可能的目录结构
        images_dir = dataset_path / "train" / "images"
        labels_dir = dataset_path / "train" / "labels"
        print(f"尝试备用目录: {images_dir}")

    if not images_dir.exists() or not labels_dir.exists():
        print(f"错误: 未找到标准的YOLO目录结构")
        print(f"图像目录: {images_dir} - {'存在' if images_dir.exists() else '不存在'}")
        print(f"标注目录: {labels_dir} - {'存在' if labels_dir.exists() else '不存在'}")
        return {}

    print(f"使用图像目录: {images_dir}")
    print(f"使用标注目录: {labels_dir}")

    # 收集所有图像文件
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))

    if not image_files:
        print(f"错误: 在 {images_dir} 中未找到图像文件")
        return {}

    print(f"找到 {len(image_files)} 张图像")

    # 按类别存储符合条件的样本
    class_samples = {}

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            # print(f"警告: 未找到标注文件 {label_path}")
            continue

        # 读取标注文件
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"错误: 读取标注文件 {label_path} 失败: {e}")
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])

                    # 检查是否在中心区域
                    if (0.5 - center_ratio / 2 <= x_center <= 0.5 + center_ratio / 2 and
                            0.5 - center_ratio / 2 <= y_center <= 0.5 + center_ratio / 2):

                        if class_id not in class_samples:
                            class_samples[class_id] = []

                        # 如果还没达到所需样本数，添加这个样本
                        if len(class_samples[class_id]) < num_samples:
                            class_samples[class_id].append({
                                'image_path': str(img_path),
                                'label_path': str(label_path),
                                'bbox': [x_center, y_center, float(parts[3]), float(parts[4])],
                                'image_name': img_path.name
                            })
                            print(f"为类别 {class_id} 找到中心样本: {img_path.name}")

                except ValueError as e:
                    print(f"错误: 解析标注行失败 {line}: {e}")
                    continue

    # 输出结果统计
    print("\n" + "=" * 50)
    print("选择结果统计:")
    print("=" * 50)
    total_samples = 0
    for class_id, samples in sorted(class_samples.items()):
        print(f"类别 {class_id}: {len(samples)} 个中心样本")
        total_samples += len(samples)
        for i, sample in enumerate(samples):
            print(f"  {i + 1}. {sample['image_name']}")

    print(f"\n总计: {total_samples} 个样本分布在 {len(class_samples)} 个类别中")

    return class_samples


# 使用示例
if __name__ == "__main__":
    # 数据集路径 - 根据你的主文件，应该使用这个路径
    dataset_path = r"C:\obb\ultralytics-yolo11-main"

    print("开始筛选中心样本...")

    # 调用筛选函数
    selected_samples = select_centered_yolo_samples(
        dataset_path,
        num_samples=5,
        center_ratio=0.3
    )

    # 设置输出文件路径 - 保存到你的项目目录下
    output_dir = r"C:\Users\20910\Desktop\selected_center_samples"
    output_file = os.path.join(output_dir, "selected_samples.json")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 将筛选结果保存到JSON文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(selected_samples, f, ensure_ascii=False, indent=2)

    print(f"\n筛选结果已保存到: {output_file}")

    # 同时创建一个简单的文本报告
    report_file = os.path.join(output_dir, "selection_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("中心样本选择报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"数据集路径: {dataset_path}\n")
        f.write(f"总类别数: {len(selected_samples)}\n")
        f.write(f"总样本数: {sum(len(samples) for samples in selected_samples.values())}\n\n")

        for class_id, samples in sorted(selected_samples.items()):
            f.write(f"类别 {class_id}:\n")
            for i, sample in enumerate(samples):
                f.write(f"  {i + 1}. {sample['image_name']}\n")
            f.write("\n")

    print(f"详细报告已保存到: {report_file}")