#%%
import json
import pandas as pd

# 文件路径
amazon_file_path = '/media/zcai/Task/CBEC_AI/Cross-Cultural Merchandising Expert/scraperAPI_samples/Amazon products.json'
shopee_file_path = '/media/zcai/Task/CBEC_AI/Cross-Cultural Merchandising Expert/scraperAPI_samples/Shopee - products.json'
tiktok_file_path = '/media/zcai/Task/CBEC_AI/Cross-Cultural Merchandising Expert/scraperAPI_samples/TikTok Shop.json'

# 加载 Amazon 数据
with open(amazon_file_path, 'r') as f:
    amazon_data = json.load(f)
amazon_df = pd.DataFrame(amazon_data)

# 加载 Shopee 数据
with open(shopee_file_path, 'r') as f:
    shopee_data = json.load(f)
shopee_df = pd.DataFrame(shopee_data)

# 加载 TikTok 数据
with open(tiktok_file_path, 'r') as f:
    tiktok_data = json.load(f)
tiktok_df = pd.DataFrame(tiktok_data)

# 获取 Amazon 数据类型
amazon_dtypes = amazon_df.dtypes

# 获取 Shopee 数据类型
shopee_dtypes = shopee_df.dtypes

# 获取 TikTok 数据类型
tiktok_dtypes = tiktok_df.dtypes

# 创建对比表格
comparison_table = pd.DataFrame({
    'Amazon Data Type': amazon_dtypes,
    'Shopee Data Type': shopee_dtypes,
    'TikTok Data Type': tiktok_dtypes
})

# 打印对比表格
print(comparison_table)

# %%
# 关键词列表
keywords = ['title', 'description', 'review', 'comment', 
            'specification', 'spec', 'detail', 'rating']

# 筛选包含关键词的数据类型
def filter_dtypes(df, keywords):
    filtered_dtypes = {}
    for col in df.columns:
        for keyword in keywords:
            if keyword in col.lower():  # 转换为小写进行不区分大小写的匹配
                filtered_dtypes[col] = df[col].dtype
                break  # 找到一个关键词就停止，避免重复添加
    return pd.Series(filtered_dtypes)

amazon_dtypes_filtered = filter_dtypes(amazon_df, keywords)
shopee_dtypes_filtered = filter_dtypes(shopee_df, keywords)
tiktok_dtypes_filtered = filter_dtypes(tiktok_df, keywords)

# 创建对比表格
comparison_table = pd.DataFrame({
    'Amazon Data Type': amazon_dtypes_filtered,
    'Shopee Data Type': shopee_dtypes_filtered,
    'TikTok Data Type': tiktok_dtypes_filtered
})

# 打印对比表格
print(comparison_table)