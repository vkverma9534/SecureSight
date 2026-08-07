# Exploratory Data Analysis

- Total number of rows = **9,516,837**

## Columns not important for analysis

- **Id** – Pure row identifier
- **AlertId** – Unique alert identifier
- **AccountSid** – User SID
- **AccountObjectId** – Azure object ID
- **NetworkMessageId** – Message ID
- **OAuthApplicationId** – Identifier only

## Target Columns

- `IncidentGrade`
- `ActionGrouped`
- `ActionGranular`

(All are Object type.)

## Input Columns

```python
[
    'OrgId', 'IncidentId', 'Timestamp', 'DetectorId', 'AlertTitle',
    'Category', 'MitreTechniques', 'EntityType', 'EvidenceRole',
    'Sha256', 'IpAddress', 'Url', 'AccountUpn', 'AccountName',
    'DeviceName', 'EmailClusterId', 'RegistryKey',
    'RegistryValueName', 'RegistryValueData', 'ApplicationName',
    'ThreatFamily', 'FileName', 'FolderPath', 'ResourceType',
    'Roles', 'OSFamily', 'OSVersion', 'AntispamDirection',
    'SuspicionLevel', 'LastVerdict', 'CountryCode',
    'State', 'City', 'DeviceId', 'ApplicationId',
    'ResourceIdName'
]
```

```python
numeric_cols = chunk_dict[0].select_dtypes(include=np.number).columns.tolist()
categorical_cols = chunk_dict[0].select_dtypes(include=object).columns.tolist()
datetime_cols = chunk_dict[0].select_dtypes(
    include=["datetime", "datetimetz"]
).columns.tolist()
```

## Numeric columns

> Initial speculation only. Many (or all) of these may actually be categorical columns stored using label encoding. Each feature will be analyzed separately.

```text
OrgId
IncidentId
DetectorId
AlertTitle
Sha256
IpAddress
Url
AccountUpn
AccountName
DeviceName
EmailClusterId
RegistryKey
RegistryValueName
RegistryValueData
ApplicationName
FileName
FolderPath
OSFamily
OSVersion
CountryCode
State
City
DeviceId
ApplicationId
```

## Categorical columns

```text
Category
MitreTechniques
EntityType
EvidenceRole
ThreatFamily
ResourceType
Roles
AntispamDirection
SuspicionLevel
LastVerdict
IncidentGrade
ActionGrouped
ActionGranular
ResourceIdName
```

## Datetime columns

```text
Timestamp
```

### Note

Columns like `City`, `CountryCode`, etc. are categorical but have high cardinality. One-hot encoding will unnecessarily expand the data.

Also, label encoding (the current representation) may mislead models since these are categorical values and nearby encoded values do not imply similarity.

Hence, instead of relying only on `select_dtypes()`, each feature will be checked individually before deciding whether it is truly numeric.

# Feature-wise EDA

## OrgId

- No need for standardization.
- No missing values.
- Data type is `int64` in every chunk.

```python
for i in range(len(chunk_dict)):
    s = chunk_dict[i]["OrgId"]
    non_numeric = s[~s.apply(lambda x: isinstance(x, (int, float)))]
    print(non_numeric.unique())
```

### Cardinality

- Cardinality is not very high considering the dataset size.
- Number of unique values varies slightly across chunks.
- Taking the union across all chunks gives **5769 unique OrgId values** in the training data.

```python
for i in range(len(chunk_dict)):
    print(chunk_dict[i]["OrgId"].nunique())
```

### Frequency Distribution

```python
freq = (
    pd.concat([chunk_dict[i]['OrgId'] for i in chunk_dict])
    .value_counts(dropna=False)
    .rename('count')
    .reset_index()
)

freq.columns = ['OrgId', 'count']
freq['count_pct'] = (
    (freq['count'] / total_valid_rows) * 100
).round(2).astype(str) + '%'
```

#### Top 20 OrgIds

| OrgId | Count | Count (%) |
|------:|------:|----------:|
| 0 | 844789 | 8.90% |
| 2 | 228325 | 2.41% |
| 1 | 210044 | 2.21% |
| 3 | 190866 | 2.01% |
| 5 | 173431 | 1.83% |
| 6 | 161092 | 1.70% |
| 4 | 145741 | 1.54% |
| 7 | 134532 | 1.42% |
| 8 | 133637 | 1.41% |
| 10 | 133160 | 1.40% |
| 9 | 130807 | 1.38% |
| 11 | 116134 | 1.22% |
| 12 | 114799 | 1.21% |
| 14 | 112681 | 1.19% |
| 13 | 107259 | 1.13% |
| 16 | 87836 | 0.93% |
| 25 | 83539 | 0.88% |
| 17 | 81424 | 0.86% |
| 19 | 80168 | 0.84% |
| 18 | 78881 | 0.83% |

### Observations

- 1845 categories appear fewer than 10 times.
- 1123 categories appear fewer than 5 times.
- Top 25 categories contribute around 40% of the data.
- Top 50 categories contribute more than 50%.
- Total unique categories = **5769**.

| Appearance | Categories |
|-----------:|-----------:|
| 10000+ | 156 |
| 1000-10000 | 600 |
| 100-1000 | 1233 |
| 10-100 | 1935 |
| 5-10 | 722 |
| 3-5 | 911 |
| 1-3 | 212 |

![Frequency Distribution](images/freq_distri_org_id.png)

![Contribution Distribution](images/contribution1.png)

- Nearly 25 Orgs appear on ~40% of the data

![Contribution Distribution](images/skewed1.png)

- We can observe how thin the line gets even when we plotted it on top 50 proving the skewness of the data

### Target Distribution

```python
dfs = []

for i in range(len(chunk_dict)):
    df = chunk_dict[i][['OrgId', 'IncidentGrade']].copy()

    category_counts = (
        df.groupby(['OrgId', 'IncidentGrade'])
          .size()
          .unstack(fill_value=0)
    )

    dfs.append(category_counts)

result = (
    pd.concat(dfs)
      .groupby('OrgId')
      .sum()
      .reset_index()
)

result.columns.name = None
result['total_count']=result['FalsePositive']+result['BenignPositive']+result['TruePositive']
result['FalsePositive_pct']=((result['FalsePositive']/result['total_count'])*100).round(3)
result['BenignPositive_pct']=((result['BenignPositive']/result['total_count'])*100).round(3)
result['TruePositive_pct']=((result['TruePositive']/result['total_count'])*100).round(3)
```
#### Conditional Target Distribution (Target | OrgId)
- This analysis examines the conditional distribution of the target variable given the feature (P(Target | OrgId)). It helps identify organizations whose incident labels are highly skewed toward a particular class, which can indicate that the feature has predictive power.

##### BenignPositive
- Organizations with more than 10,000 incidents were ranked by their BenignPositive percentage. Several OrgIds exhibit a very high BenignPositive rate, suggesting a strong relationship between OrgId and the target variable.

```python
df=result[['OrgId','total_count','FalsePositive_pct','BenignPositive_pct','TruePositive_pct']][result['total_count']>10000]
BenignPositive_df=df.sort_values('BenignPositive_pct',ascending=False)
BenignPositive_df.head(20)
```
| OrgId | Total Count | FalsePositive (%) | BenignPositive (%) | TruePositive (%) |
|------:|------------:|------------------:|-------------------:|-----------------:|
| 21 | 73,390 | 0.000 | **100.000** | 0.000 |
| 25 | 83,531 | 0.000 | **100.000** | 0.000 |
| 120 | 13,382 | 0.000 | **100.000** | 0.000 |
| 161 | 10,070 | 0.000 | **100.000** | 0.000 |
| 51 | 40,132 | 0.000 | **100.000** | 0.000 |
| 12 | 114,799 | 0.014 | 99.986 | 0.000 |
| 13 | 107,259 | 0.021 | 99.979 | 0.000 |
| 3 | 190,866 | 0.013 | 99.977 | 0.010 |
| 83 | 21,209 | 0.038 | 99.962 | 0.000 |
| 31 | 58,276 | 0.041 | 99.959 | 0.000 |
| 2 | 228,325 | 0.020 | 99.958 | 0.022 |
| 133 | 12,658 | 0.063 | 99.937 | 0.000 |
| 16 | 87,836 | 0.065 | 99.935 | 0.000 |
| 64 | 30,635 | 0.049 | 99.925 | 0.026 |
| 44 | 44,096 | 0.061 | 99.850 | 0.088 |
| 102 | 21,421 | 0.019 | 99.804 | 0.177 |
| 149 | 11,084 | 0.000 | 99.783 | 0.217 |
| 40 | 55,287 | 0.127 | 99.745 | 0.128 |
| 48 | 43,506 | 0.064 | 99.549 | 0.386 |
| 68 | 25,608 | 0.223 | 99.375 | 0.402 |

##### FalsePositive
- Organizations with more than 10,000 incidents were ranked by their FalsePositive percentage. Several OrgIds exhibit a very high FalsePositive rate, suggesting a strong relationship between OrgId and the target variable.
```python
FalsePositive_df=FalsePositive_df.sort_values('FalsePositive_pct',ascending=False)
FalsePositive_df.head(20)
```

| OrgId | Total Count | FalsePositive (%) | BenignPositive (%) | TruePositive (%) |
|------:|------------:|------------------:|-------------------:|-----------------:|
| 60 | 23,328 | **100.000** | 0.000 | 0.000 |
| 88 | 24,244 | **100.000** | 0.000 | 0.000 |
| 140 | 10,832 | **100.000** | 0.000 | 0.000 |
| 37 | 64,702 | **100.000** | 0.000 | 0.000 |
| 79 | 22,131 | 99.991 | 0.000 | 0.009 |
| 71 | 30,902 | 99.990 | 0.000 | 0.010 |
| 17 | 81,424 | 99.989 | 0.000 | 0.011 |
| 119 | 17,343 | 99.983 | 0.000 | 0.017 |
| 142 | 12,503 | 99.968 | 0.000 | 0.032 |
| 187 | 10,584 | 99.943 | 0.000 | 0.057 |
| 47 | 44,531 | 99.879 | 0.115 | 0.007 |
| 104 | 18,985 | 99.768 | 0.232 | 0.000 |
| 18 | 78,870 | 99.654 | 0.000 | 0.346 |
| 124 | 14,009 | 99.557 | 0.000 | 0.443 |
| 98 | 17,387 | 98.763 | 0.092 | 1.145 |
| 159 | 10,500 | 96.695 | 0.257 | 3.048 |
| 11 | 116,134 | 96.469 | 0.000 | 3.531 |
| 7 | 134,532 | 94.963 | 3.631 | 1.406 |
| 122 | 14,854 | 93.611 | 1.299 | 5.090 |
| 87 | 22,213 | 89.614 | 10.003 | 0.383 |

##### TruePositive
- Organizations with more than 10,000 incidents were ranked by their TruePositive percentage. Several OrgIds exhibit a very high TruePositive rate, suggesting a strong relationship between OrgId and the target variable.
```python
TruePositive_df=df.sort_values('TruePositive_pct',ascending=False)
TruePositive_df.head(20)
```

| OrgId | Total Count | FalsePositive (%) | BenignPositive (%) | TruePositive (%) |
|------:|------------:|------------------:|-------------------:|-----------------:|
| 22 | 60,418 | 0.000 | 0.000 | **100.000** |
| 72 | 25,866 | 0.000 | 0.000 | **100.000** |
| 110 | 13,545 | 0.000 | 0.000 | **100.000** |
| 63 | 27,288 | 0.000 | 0.000 | **100.000** |
| 76 | 22,631 | 0.000 | 0.000 | **100.000** |
| 0 | 844,782 | 0.005 | 0.000 | 99.995 |
| 117 | 19,701 | 0.020 | 0.000 | 99.980 |
| 35 | 46,807 | 0.043 | 0.000 | 99.957 |
| 148 | 13,856 | 0.058 | 0.000 | 99.942 |
| 89 | 11,262 | 0.098 | 0.000 | 99.902 |
| 1 | 210,035 | 0.103 | 0.000 | 99.897 |
| 90 | 18,579 | 0.151 | 0.000 | 99.849 |
| 169 | 12,921 | 0.217 | 0.000 | 99.783 |
| 10 | 133,160 | 0.000 | 0.271 | 99.729 |
| 41 | 53,993 | 0.267 | 0.035 | 99.698 |
| 75 | 33,901 | 0.342 | 0.000 | 99.658 |
| 178 | 12,188 | 0.361 | 0.000 | 99.639 |
| 8 | 133,629 | 0.058 | 0.542 | 99.400 |
| 5 | 173,409 | 0.607 | 0.000 | 99.393 |
| 101 | 18,463 | 0.628 | 0.000 | 99.372 |

#### Dominant Class Category
```python
target_cols = ['FalsePositive', 'BenignPositive', 'TruePositive']
result['max_target_pct'] = result[target_cols].max(axis=1) / result['total_count'] * 100
result['max_target_pct'].describe()
plt.figure(figsize=(8,5))
plt.hist(result['max_target_pct'], bins=30)
plt.xlabel("Dominant Class Percentage")
plt.ylabel("Number of OrgIds")
plt.title("Distribution of Dominant Target Percentage")
plt.show()
```

![Dominant Class Category](images/dom_class1.png)

#### Purity Statistics

100% Pure : 3047
>99% Pure : 3208
>95% Pure : 3530
<80% Pure : 1103

#### Entrory Distribution

```python
p = result[target_cols].div(result['total_count'], axis=0)
result['entropy'] = -(p * np.log2(p.replace(0, np.nan))).sum(axis=1)
result[['OrgId','entropy']].head()
plt.figure(figsize=(8,5))
plt.hist(result['entropy'], bins=30)
plt.xlabel("Entropy")
plt.ylabel("Number of OrgIds")
plt.title("Entropy Distribution Across OrgIds")
plt.show()
```

![Entropy Distribution](images/entropy_dist.png)

#### Target Distribtution for top 20 Orgs
```python
top = result.nlargest(20,'total_count')

plot_df = top.set_index('OrgId')[
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
plt.title("Target Distribution for Top 20 OrgIds")
plt.show()
```

![Target Distribution For top 20 orgs](images/tar_dist_top_20.png)

#### Dominant Class Count
```python
result['majority_class'].value_counts().plot.bar()
plt.ylabel("Number of OrgIds")
plt.title("Dominant IncidentGrade per OrgId")
plt.show()
```
![Dominant Class Count](images/dom_class_count.png)

### Relationship with other numeric Cols
```python
corr_list = []

for chunk in chunk_dict.values():
    corr_list.append(chunk[numeric_cols].corr()['OrgId'])

org_corr = pd.concat(corr_list, axis=1).mean(axis=1)
org_corr = org_corr.drop('OrgId').sort_values(key=abs, ascending=False)

print(org_corr)
```
| Feature | Correlation with OrgId |
|:------------------|-----------------------:|
| DetectorId | 0.166254 |
| IpAddress | 0.116416 |
| ApplicationName | -0.056599 |
| ApplicationId | -0.052228 |
| DeviceName | 0.040555 |
| AccountName | -0.038584 |
| City | 0.027538 |
| FolderPath | 0.026376 |
| AccountUpn | -0.026347 |
| IncidentId | 0.025984 |
| CountryCode | 0.025942 |
| EmailClusterId | -0.025336 |
| State | 0.025314 |
| RegistryValueData | -0.015301 |
| FileName | 0.011773 |
| AlertTitle | 0.011576 |
| Url | 0.011563 |
| RegistryValueName | -0.011410 |
| OSVersion | -0.011312 |
| OSFamily | -0.010892 |
| DeviceId | 0.007700 |
| Sha256 | 0.002955 |
| RegistryKey | -0.002926 |

- Observation: OrgId exhibits generally weak linear correlations with the remaining numeric (identifier) features. The highest positive correlation is with DetectorId (0.166), followed by IpAddress (0.116), while all other correlations are close to zero (|r| < 0.06). This suggests that OrgId captures information largely independent of the other encoded identifier features. Since these variables are high-cardinality identifiers rather than true continuous numeric measurements, Pearson correlation should be interpreted only as a descriptive measure and not as evidence of strong feature dependence