import pandas as pd
import numpy as np
from scipy import stats

# 1. 读取两张CSV文件
your_model_csv = "SAM-newAdapter-decoder-otherMethods/testW/mynet.csv"  # 你的模型CSV
ref_model_csv = "SAM-newAdapter-decoder-otherMethods/testW/emacd.csv"  # 对比模型CSV

your_df = pd.read_csv(your_model_csv)
ref_df = pd.read_csv(ref_model_csv)

# 2. 核心修改：只保留双方都存在的图片（按image_name取交集）
# 提取两张表的图片名称集合
your_image_names = set(your_df['image_name'].values)
ref_image_names = set(ref_df['image_name'].values)

# 求交集：获取双方都存在的图片名称
common_image_names = your_image_names & ref_image_names
print(f"双方共同存在的图片数量：{len(common_image_names)}")
print(f"你的模型独有的图片数量：{len(your_image_names - common_image_names)}")
print(f"对比模型独有的图片数量：{len(ref_image_names - common_image_names)}")

# 按共同图片名称筛选两张表（保持图片顺序一致）
your_df_common = your_df[your_df['image_name'].isin(common_image_names)].sort_values(by='image_name').reset_index(drop=True)
ref_df_common = ref_df[ref_df['image_name'].isin(common_image_names)].sort_values(by='image_name').reset_index(drop=True)

# 提取筛选后的challenge IoU数组
your_image_iou = your_df_common['challenge_iou'].values
ref_image_iou = ref_df_common['challenge_iou'].values

# 3. 验证筛选后的数据一致性（可选，确保无问题）
assert len(your_image_iou) == len(ref_image_iou), "筛选后两张CSV的图片数量仍不一致！"
assert (your_df_common['image_name'] == ref_df_common['image_name']).all(), "筛选后两张CSV的图片顺序/名称不一致！"
print(f"筛选后有效配对样本数量：{len(your_image_iou)}")

# 4. 数据校验与容错处理
diff = your_image_iou - ref_image_iou
if (diff == 0).all():
    print("⚠️  警告：两组数据完全一致，差值全为0，无法执行有效Wilcoxon检验")
    wilcox_w = np.nan
    p_val = 1.0
else:
    # 执行Wilcoxon符号秩检验（单侧，判断你的模型更优）
    try:
        wilcox_w, p_val = stats.wilcoxon(
            x=your_image_iou,
            y=ref_image_iou,
            alternative='greater',
            zero_method='wilcox'
        )
    except Exception as e:
        print(f"⚠️  执行Wilcoxon检验失败：{str(e)}")
        wilcox_w = np.nan
        p_val = np.nan

# 5. 输出结果
print("\n=== Wilcoxon 符号秩检验结果 ===")
if not np.isnan(wilcox_w):
    print(f"Wilcoxon W统计量: {wilcox_w:.4f}")
else:
    print(f"Wilcoxon W统计量: 无效（数据无差异或检验失败）")

if not np.isnan(p_val):
    print(f"P值: {p_val:.6f}")
else:
    print(f"P值: 无效（数据无差异或检验失败）")

# 计算并输出平均IoU差值
mean_iou_diff = np.mean(your_image_iou) - np.mean(ref_image_iou)
print(f"你的模型平均challenge IoU - 对比模型平均challenge IoU: {mean_iou_diff:.4f}")