## RegistryKey

- No need for standardization.
- No missing values.
- Data type is `int64` in every chunk.

```python
for i in range(len(chunk_dict)):
    s = chunk_dict[i]["RegistryKey"]
    non_numeric = s[~s.apply(lambda x: isinstance(x, (int, float)))]
    print(non_numeric.unique())
```

### Cardinality

- Cardinality is very high considering the dataset size.
- Number of unique values varies highly across chunks each chunks appears to be containing ~20000 uniques values of RegistryKey 5 times lower than total nuniques.
- Taking the union across all chunks gives **106416 unique RegistryKey values** in the training data.

```python
for i in range(len(chunk_dict)):
    print(chunk_dict[i]["RegistryKey"].nunique())
```

### Frequency Distribution

```python
freq = (
    pd.concat([chunk_dict[i]['RegistryKey'] for i in chunk_dict])
    .value_counts(dropna=False)
    .rename('count')
    .reset_index()
)
freq=freq.sort_values('count',ascending=False)
freq['count_pct(%)']=((freq['count']/9516837)*100).round(3)
freq['cumsum_pct']=freq['count_pct(%)'].cumsum()
freq.columns = ['RegistryKey', 'count','count_pct(%)','cumsum_pct']
```

#### Top 50 RegistryKeys

| RegistryKey | count | count_pct(%) | cumsum_pct |
|---:|---:|---:|---:|
| 138268 | 8,781,572 | 92.274 | 92.274 |
| 0 | 50,483 | 0.530 | 92.804 |
| 1 | 27,592 | 0.290 | 93.094 |
| 2 | 15,918 | 0.167 | 93.261 |
| 3 | 15,306 | 0.161 | 93.422 |
| 4 | 11,444 | 0.120 | 93.542 |
| 5 | 8,545 | 0.090 | 93.632 |
| 6 | 7,579 | 0.080 | 93.712 |
| 7 | 7,469 | 0.078 | 93.790 |
| 8 | 7,413 | 0.078 | 93.868 |
| 12 | 5,058 | 0.053 | 93.921 |
| 10 | 5,015 | 0.053 | 93.974 |
| 11 | 4,964 | 0.052 | 94.026 |
| 9 | 4,944 | 0.052 | 94.078 |
| 13 | 4,403 | 0.046 | 94.124 |
| 15 | 4,080 | 0.043 | 94.167 |
| 14 | 4,056 | 0.043 | 94.210 |
| 16 | 3,585 | 0.038 | 94.248 |
| 19 | 3,173 | 0.033 | 94.281 |
| 17 | 2,742 | 0.029 | 94.310 |
| 18 | 2,609 | 0.027 | 94.337 |
| 20 | 2,469 | 0.026 | 94.363 |
| 21 | 2,412 | 0.025 | 94.388 |
| 23 | 2,101 | 0.022 | 94.410 |
| 34 | 2,086 | 0.022 | 94.432 |
| 24 | 2,055 | 0.022 | 94.454 |
| 22 | 2,051 | 0.022 | 94.476 |
| 26 | 1,914 | 0.020 | 94.496 |
| 29 | 1,910 | 0.020 | 94.516 |
| 28 | 1,910 | 0.020 | 94.536 |
| 30 | 1,910 | 0.020 | 94.556 |
| 31 | 1,910 | 0.020 | 94.576 |
| 27 | 1,882 | 0.020 | 94.596 |
| 25 | 1,810 | 0.019 | 94.615 |
| 33 | 1,810 | 0.019 | 94.634 |
| 32 | 1,706 | 0.018 | 94.652 |
| 37 | 1,696 | 0.018 | 94.670 |
| 39 | 1,693 | 0.018 | 94.688 |
| 38 | 1,634 | 0.017 | 94.705 |
| 35 | 1,627 | 0.017 | 94.722 |
| 36 | 1,627 | 0.017 | 94.739 |
| 42 | 1,599 | 0.017 | 94.756 |
| 40 | 1,541 | 0.016 | 94.772 |
| 41 | 1,502 | 0.016 | 94.788 |
| 44 | 1,493 | 0.016 | 94.804 |
| 46 | 1,394 | 0.015 | 94.819 |
| 51 | 1,377 | 0.014 | 94.833 |
| 45 | 1,343 | 0.014 | 94.847 |
| 47 | 1,336 | 0.014 | 94.861 |
| 55 | 1,314 | 0.014 | 94.875 |

### Observations

- 99831 categories appear fewer than 10 times.
- 90758 categories appear fewer than 5 times.
- Top 1 category contribute around 92% of the data.
- Top 50 categories contribute more than 94%.
- Total unique categories = **106416**.

| Appearance | Categories |
|-----------:|-----------:|
| 10000+ | 6 |
| 1000-10000 | 52 |
| 100-1000 | 553 |
| 10-100 | 5974 |
| 5-10 | 9073 |
| 3-5 | 16185 |
| 1-3 | 74575 |

![Frequency Distribution](images/freq_distri_RegistryKey.png)

- Nearly 1 RegistryKeys appear on ~92% of the data

![Contribution Distribution](images/skewedRegistryKey.png)

- We can observe how thin the line gets even when we plotted it on top 50 proving the skewness of the data, only 1 category has almost complete participation and then a sudden drop

### Target Distribution

```python
dfs = []

for i in range(len(chunk_dict)):
    df = chunk_dict[i][['RegistryKey', 'IncidentGrade']].copy()

    category_counts = (
        df.groupby(['RegistryKey', 'IncidentGrade'])
          .size()
          .unstack(fill_value=0)
    )

    dfs.append(category_counts)

result = (
    pd.concat(dfs)
      .groupby('RegistryKey')
      .sum()
      .reset_index()
)

result.columns.name = None
result['total_count']=result['FalsePositive']+result['BenignPositive']+result['TruePositive']
result['FalsePositive_pct']=((result['FalsePositive']/result['total_count'])*100).round(3)
result['BenignPositive_pct']=((result['BenignPositive']/result['total_count'])*100).round(3)
result['TruePositive_pct']=((result['TruePositive']/result['total_count'])*100).round(3)
```
#### Conditional Target Distribution (Target | RegistryKey)
- This analysis examines the conditional distribution of the target variable given the feature (P(Target | RegistryKey)). It helps identify RegistryKey whose incident labels are highly skewed toward a particular class, which can indicate that the feature has predictive power.

##### BenignPositive
- RegistryKey with more than 10,000 incidents were ranked by their BenignPositive percentage. Several RegistryKeys exhibit a very high BenignPositive rate, suggesting a strong relationship between RegistryKey and the target variable.

```python
df=result[['RegistryKey','total_count','FalsePositive_pct','BenignPositive_pct','TruePositive_pct']][result['total_count']>10000]
BenignPositive_df=df.sort_values(by=['BenignPositive_pct','total_count'],ascending=[False,False])
BenignPositive_df.head(20)
```
| RegistryKey | total_count | FalsePositive_pct | BenignPositive_pct | TruePositive_pct |
|---:|---:|---:|---:|---:|
| 4 | 11,444 | 0.105 | **99.878** | 0.017 |
| 3 | 15,306 | 11.930 | 78.374 | 9.696 |
| 2 | 15,918 | 11.616 | 76.165 | 12.219 |
| 1 | 27,592 | 12.399 | 74.768 | 12.833 |
| 0 | 50,483 | 14.742 | 64.598 | 20.660 |
| 138268 | 8,730,233 | 21.732 | 41.505 | 36.764 |

##### FalsePositive
- RegistryKey with more than 10,000 incidents were ranked by their FalsePositive percentage. Several RegistryKeys exhibit a very high FalsePositive rate, suggesting a strong relationship between RegistryKey and the target variable.
```python
FalsePositive_df=df.sort_values(by=['FalsePositive_pct','total_count'],ascending=[False,False])
FalsePositive_df.head(20)
```

| RegistryKey | total_count | FalsePositive_pct | BenignPositive_pct | TruePositive_pct |
|---:|---:|---:|---:|---:|
| 138268 | 8,730,233 | **21.732** | 41.505 | 36.764 |
| 0 | 50,483 | 14.742 | 64.598 | 20.660 |
| 1 | 27,592 | 12.399 | 74.768 | 12.833 |
| 3 | 15,306 | 11.930 | 78.374 | 9.696 |
| 2 | 15,918 | 11.616 | 76.165 | 12.219 |
| 4 | 11,444 | 0.105 | **99.878** | 0.017 |

##### TruePositive
- RegistryKey with more than 10,000 incidents were ranked by their TruePositive percentage. Several RegistryKeys exhibit a very high TruePositive rate, suggesting a strong relationship between RegistryKey and the target variable.
```python
TruePositive_df=df.sort_values(by=['TruePositive_pct','total_count'],ascending=[False,False])
TruePositive_df.head(20)
```
| RegistryKey | total_count | FalsePositive_pct | BenignPositive_pct | TruePositive_pct |
|---:|---:|---:|---:|---:|
| 138268 | 8,730,233 | **21.732** | 41.505 | 36.764 |
| 0 | 50,483 | 14.742 | 64.598 | 20.660 |
| 1 | 27,592 | 12.399 | 74.768 | 12.833 |
| 3 | 15,306 | 11.930 | 78.374 | 9.696 |
| 2 | 15,918 | 11.616 | 76.165 | 12.219 |
| 4 | 11,444 | 0.105 | **99.878** | 0.017 |

#### Dominant Class Category
```python
target_cols = ['FalsePositive', 'BenignPositive', 'TruePositive']
result['max_target_pct'] = result[target_cols].max(axis=1) / result['total_count'] * 100
result['max_target_pct'].describe()
plt.figure(figsize=(8,5))
plt.hist(result['max_target_pct'], bins=30)
plt.xlabel("Dominant Class Percentage")
plt.ylabel("Number of RegistryKeys")
plt.title("Distribution of Dominant Target Percentage")
plt.show()
```

![Dominant Class Category](images/dom_class_RegistryKey.png)

#### Purity Statistics
```python
print("100% Pure :", (result['max_target_pct'] == 100).sum())
print(">99% Pure :", (result['max_target_pct'] >= 99).sum())
print(">95% Pure :", (result['max_target_pct'] >= 95).sum())
print("<80% Pure :", (result['max_target_pct'] < 80).sum())
```
100% Pure : 102362
>99% Pure : 102381
>95% Pure : 102531
<80% Pure : 3070

#### Entrory Distribution

```python
p = result[target_cols].div(result['total_count'], axis=0)
result['entropy'] = -(p * np.log2(p.replace(0, np.nan))).sum(axis=1)
result[['RegistryKey','entropy']].head()
plt.figure(figsize=(8,5))
plt.hist(result['entropy'], bins=30)
plt.xlabel("Entropy")
plt.ylabel("Number of RegistryKeys")
plt.title("Entropy Distribution Across RegistryKeys")
plt.show()
```

![Entropy Distribution](images/entropy_dist_RegistryKey.png)

#### Target Distribtution for top 20 RegistryKeys
```python
top = result.nlargest(20,'total_count')

plot_df = top.set_index('RegistryKey')[
    ['FalsePositive_pct',
     'BenignPositive_pct',
     'TruePositive_pct']
]
plot_df.plot(
    kind='bar',
    stacked=True,
    figsize=(12,5)
)
plt.ylabel("Percentage")
plt.title("Target Distribution for Top 20 RegistryKeys")
plt.show()
```

![Target Distribution For top 20 RegistryKeys](images/tar_dist_top_20_RegistryKey.png)

#### Dominant Class Count
```python
result['majority_class'].value_counts().plot.bar()
plt.ylabel("Number of RegistryKeys")
plt.title("Dominant IncidentGrade per RegistryKey")
plt.show()
```
![Dominant Class Count](images/dom_class_count_RegistryKey.png)

### Relationship with other numeric Cols
```python
corr_list = []

for chunk in chunk_dict.values():
    corr_list.append(chunk[numeric_cols].corr()['RegistryKey'])

RegistryKey_corr = pd.concat(corr_list, axis=1).mean(axis=1)
RegistryKey_corr = RegistryKey_corr.drop('RegistryKey').sort_values(key=abs, ascending=False)

print(RegistryKey_corr)
```
| Feature | Correlation |
|---|---:|
| **FileName** | **0.864043** |
| **FolderPath** | **0.751187** |
| AccountUpn | -0.202853 |
| AccountName | -0.155127 |
| IpAddress | -0.148903 |
| CountryCode | -0.081957 |
| DeviceName | -0.076764 |
| Url | -0.075364 |
| State | -0.074819 |
| City | -0.074742 |
| DeviceId | -0.053846 |
| ApplicationName | -0.043060 |
| ApplicationId | -0.042449 |
| OSVersion | -0.040546 |
| OSFamily | -0.040517 |
| IncidentId | -0.024234 |
| DetectorId | -0.021057 |
| RegistryKey | -0.011912 |
| AlertTitle | 0.011300 |
| RegistryValueData | -0.006551 |
| RegistryValueName | -0.005926 |
| OrgId | 0.002955 |
| EmailClusterId | NaN |

- Observation: The features exhibit mixed linear correlations with the target. FileName (0.864) shows the strongest positive correlation, followed by FolderPath (0.751). AccountUpn (-0.203) has the strongest negative correlation, followed by AccountName (-0.155) and IpAddress (-0.149). Most remaining features show weak correlations close to zero. Since these features are largely categorical or high-cardinality identifiers, Pearson correlation should be interpreted as a descriptive measure only, rather than evidence of a meaningful relationship or causation.