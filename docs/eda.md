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

| OrgId | Count | % |
|------:|------:|--:|
| 0 | 844789 | 8.90 |
| 2 | 228325 | 2.41 |
| 1 | 210044 | 2.21 |
| ... | ... | ... |

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