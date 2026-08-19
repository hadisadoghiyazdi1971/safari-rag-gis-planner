import pandas as pd

# خواندن اکسل و نمایش ستون‌ها
df = pd.read_excel("African.xls")
print("ستون‌های موجود در اکسل:")
print(df.columns.tolist())
print(f"\nتعداد ستون‌ها: {len(df.columns)}")
print(f"\nنمونه داده‌ها:")
print(df.head(2))