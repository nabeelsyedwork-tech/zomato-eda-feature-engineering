import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('zomato.csv')
# print(df.head())
# print(df.tail())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

df.drop_duplicates(inplace=True)
df['rate'] = (
    df['rate']
    .replace(['NEW', '-'], np.nan)
    .astype(str)
    .str.split('/')
    .str[0]
    .astype(float)
)
df['rate'].fillna(df['rate'].mean(), inplace=True)
df['location'].fillna(df['location'].mode()[0], inplace=True)
df['cuisines'].fillna('Unknown', inplace=True)

df['approx_cost(for two people)'] = df['approx_cost(for two people)'].str.replace(',', '').astype(float)
df['Has_Online'] = (df['online_order'] == 'Yes').astype(int)
df['Has_Table'] = (df['book_table'] == 'Yes').astype(int)
df['Is_Expensive'] = (df['approx_cost(for two people)'] > df['approx_cost(for two people)'].median()).astype(int)
df['Cuisine_Count'] = df['cuisines'].str.split(',').str.len()
df['avg_rating_location'] = df.groupby('location')['rate'].transform('mean')
df['avg_rating_cuisines'] = df.groupby('cuisines')['rate'].transform('mean')

def clip_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    return series.clip(lower, upper)

df['rate'] = clip_outliers(df['rate'])
df['approx_cost(for two people)'] = clip_outliers(df['approx_cost(for two people)'])
df['votes'] = clip_outliers(df['votes'])

def group_rare(series,N):
    top_categories = series.value_counts().index[:N]
    return series.apply(lambda x: x if x in top_categories else 'Other')

all_dishes = df['dish_liked'].dropna().str.split(',').explode()
top_dishes = all_dishes.value_counts().index[:10]

def filter_top_dishes(dishes):
    if pd.isna(dishes):
        return 'Other'
    filtered = [d for d in dishes.split(',') if d in top_dishes]
    return ','.join(filtered) if filtered else 'Other'

df['dish_liked_grouped'] = df['dish_liked'].apply(filter_top_dishes)
df['rest_type'] = group_rare(df['rest_type'],10)
df['location'] = group_rare(df['location'],10)

# sns.histplot(df['rate'], bins=20, kde=True, color='orange')
# plt.title("Distribution of Restaurant Ratings")
# plt.show()
# sns.boxplot(x='online_order', y='rate', data=df, palette='Set2')
# plt.title("Rating Distribution by Online Order Availability")
# plt.show()
# sns.boxplot(x='book_table', y='rate', data=df, palette='Set3')
# plt.title("Ratings vs Table Booking")
# plt.show()
# top_cuisines = df['cuisines'].value_counts().head(10)
# sns.barplot(y=top_cuisines.index, x=top_cuisines.values, palette='coolwarm')
# plt.title("Top 10 Most Common Cuisines")
# plt.xlabel("Number of Restaurants")
# plt.show()
# top_locations = df['location'].value_counts().head(10)
# sns.barplot(y=top_locations.index, x=top_locations.values, palette='magma')
# plt.title("Top 10 Locations with Most Restaurants")
# plt.xlabel("Number of Restaurants")
# plt.show()
# numeric_df = df.select_dtypes(include=['float', 'int'])
# sns.heatmap(numeric_df.corr(), annot=True, cmap='viridis')
# plt.title("Correlation Between Numerical Features")
# plt.show()
scaler = StandardScaler()
df[['rate', 'approx_cost(for two people)', 'Cuisine_Count']] = scaler.fit_transform(
    df[['rate', 'approx_cost(for two people)', 'Cuisine_Count']]
)

df = pd.get_dummies(df, columns=['location', 'rest_type'], drop_first=True)
dish_dummies = df['dish_liked_grouped'].str.get_dummies(sep=',')
dish_dummies = dish_dummies[top_dishes]
df = pd.concat([df, dish_dummies], axis=1)
df = df.loc[:,~df.columns.duplicated()]
bool_cols = df.select_dtypes('bool').columns
df[bool_cols] = df[bool_cols].astype(int)
df.drop(['url','address','phone','reviews_list','menu_item'], axis=1, inplace=True)
