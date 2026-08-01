# Dataset Description

### First Impressions from the Data

 - The Training data has 9,516,837 rows
 - Since I have to do this on Google Colab because of unavailability of GPU and limited RAM and storage the data still seems be crashing the RAM given by colab
 - To solve this we come up with chunks of data each consisting a set of 800000 rows chunk size which gives us 12 chunks last chunk (chunk 11) is of size 716837 rows (we need to keep the chunks date sorted)
 - We need to make a dictionary chunk_dict={} to get full data as follows

 ```python
    chunks = pd.read_csv(
        "drive/MyDrive/GUIDE_Train.csv",
        chunksize=800_000,
        parse_dates=["Timestamp"]
    )

    chunk_dict = {}

    for i, chunk in enumerate(chunks):
        chunk = chunk.sort_values("Timestamp")
        chunk_dict[i] = chunk
```
 - Columns (45 total) in the data are following
 ```text
 Index(['Id', 'OrgId', 'IncidentId', 'AlertId', 'Timestamp', 'DetectorId',
       'AlertTitle', 'Category', 'MitreTechniques', 'IncidentGrade',
       'ActionGrouped', 'ActionGranular', 'EntityType', 'EvidenceRole',
       'DeviceId', 'Sha256', 'IpAddress', 'Url', 'AccountSid', 'AccountUpn',
       'AccountObjectId', 'AccountName', 'DeviceName', 'NetworkMessageId',
       'EmailClusterId', 'RegistryKey', 'RegistryValueName',
       'RegistryValueData', 'ApplicationId', 'ApplicationName',
       'OAuthApplicationId', 'ThreatFamily', 'FileName', 'FolderPath',
       'ResourceIdName', 'ResourceType', 'Roles', 'OSFamily', 'OSVersion',
       'AntispamDirection', 'SuspicionLevel', 'LastVerdict', 'CountryCode',
       'State', 'City'],
 ```

 - From what I can see there are many columns that are particular to the individual who raised the alert. I do not think such columns help too much in creating models that are too general for detecting vulnerabilities but to get to any concluding decision we need to analyzing each column

#### Definition for columns

*The bracket comments the importance of feature*
 - **Id** – Unique record identifier. *(Usually no)*
 - **OrgId** – Organization identifier. *(Depends)*
 - **IncidentId** – Groups related alerts into an incident. *(Usually no)*
 - **AlertId** – Unique alert identifier. *(Usually no)*
 - **Timestamp** – Time the alert was generated. *(Yes)*
 - **DetectorId** – Detection rule that generated the alert. *(Yes)*
 - **AlertTitle** – Name of the triggered detection. *(Yes)*
 - **Category** – Type of security alert. *(Yes)*
 - **MitreTechniques** – MITRE ATT&CK technique(s) detected. *(Yes)*
 - **IncidentGrade** – Triage label (target variable).
 - **ActionGrouped** – High-level remediation action (target for remediation task).
 - **ActionGranular** – Detailed remediation action (target for remediation task).
 - **EntityType** – Type of involved entity. *(Yes)*
 - **EvidenceRole** – Role of the entity in the alert. *(Yes)*
 - **DeviceId** – Unique device identifier. *(Usually no)*
 - **Sha256** – File hash. *(Sometimes)*
 - **IpAddress** – IP address involved. *(Sometimes)*
 - **Url** – URL involved. *(Sometimes)*
 - **AccountSid** – Windows account SID. *(Usually no)*
 - **AccountUpn** – User login (UPN). *(Usually no)*
 - **AccountObjectId** – Azure AD account ID. *(Usually no)*
 - **AccountName** – Username. *(Depends)*
 - **DeviceName** – Device hostname. *(Usually no)*
 - **NetworkMessageId** – Email/message identifier. *(Usually no)*
 - **EmailClusterId** – Identifier for related emails. *(Depends)*
 - **RegistryKey** – Windows registry key. *(Depends)*
 - **RegistryValueName** – Registry value name. *(Depends)*
 - **RegistryValueData** – Registry value content. *(Depends)*
 - **ApplicationId** – Application identifier. *(Usually no)*
 - **ApplicationName** – Application name. *(Depends)*
 - **OAuthApplicationId** – OAuth application identifier. *(Usually no)*
 - **ThreatFamily** – Malware/threat family. *(Yes)*
 - **FileName** – Name of the affected file. *(Sometimes)*
 - **FolderPath** – File directory path. *(Sometimes)*
 - **ResourceIdName** – Cloud resource name/ID. *(Usually no)*
 - **ResourceType** – Type of cloud resource. *(Yes)*
 - **Roles** – Account or resource roles. *(Yes)*
 - **OSFamily** – Operating system family. *(Yes)*
 - **OSVersion** – Operating system version. *(Yes)*
 - **AntispamDirection** – Email direction (inbound/outbound). *(Yes)*
 - **SuspicionLevel** – Detection confidence level. *(Yes)*
 - **LastVerdict** – Final detection verdict. *(Yes)*
 - **CountryCode** – Country associated with the activity. *(Depends)*
 - **State** – State/region. *(Depends)*
 - **City** – City. *(Depends)*

#### Action

 - We'll start with dropping columns (9) that are almost certainly not required for analysis following are those features along with reasons
  - **Id** – Pure row identifier
  - **AlertId** – Unique alert identifier
  - **AccountSid** – User SID
  - **AccountObjectId** – Azure object ID
  - **NetworkMessageId** – Message ID
  - **OAuthApplicationId** – Identifier only

Hence,
```python
input_cols = ['OrgId', 'IncidentId', 'Timestamp', 'DetectorId', 'AlertTitle',
              'Category', 'MitreTechniques', 'EntityType', 'EvidenceRole', 'Sha256',
              'IpAddress', 'Url', 'AccountUpn', 'AccountName', 'DeviceName',
              'EmailClusterId', 'RegistryKey', 'RegistryValueName',
              'RegistryValueData', 'ApplicationName', 'ThreatFamily', 'FileName',
              'FolderPath', 'ResourceType', 'Roles', 'OSFamily', 'OSVersion',
              'AntispamDirection', 'SuspicionLevel', 'LastVerdict', 'CountryCode',
              'State', 'City','DeviceId','ApplicationId','ResourceIdName']
```

and

```python
target_cols=['IncidentGrade','ActionGrouped','ActionGranular']
```

#### Target Features

 - IncidentGrade : The final triage label assigned to an alert, indicating whether it is a True Positive, Benign Positive, or False Positive.

 - ActionGrouped : The high-level remediation action recommended for handling the incident, such as Investigate, Block, or Isolate Device.

 - ActionGranular – The specific remediation step to take for the incident, providing a more detailed action than ActionGrouped (e.g., Block File Hash, Disable User Account, Kill Process).

#### Data Profiling (on first chunk to get an idea)

        column:OrgId  nunique:4262
        column:IncidentId  nunique:212393
        column:Timestamp  nunique:376662
        column:DetectorId  nunique:5306
        column:AlertTitle  nunique:42327
        column:Category  nunique:19
        column:MitreTechniques  nunique:911
        column:EntityType  nunique:28
        column:EvidenceRole  nunique:2
        column:Sha256  nunique:21238
        column:IpAddress  nunique:61885
        column:Url  nunique:22951
        column:AccountUpn  nunique:142810
        column:AccountName  nunique:96585
        column:DeviceName  nunique:25975
        column:EmailClusterId  nunique:4730
        column:RegistryKey  nunique:316
        column:RegistryValueName  nunique:126
        column:RegistryValueData  nunique:175
        column:ApplicationName  nunique:493
        column:ThreatFamily  nunique:837
        column:FileName  nunique:35872
        column:FolderPath  nunique:17688
        column:ResourceType  nunique:22
        column:Roles  nunique:9
        column:OSFamily  nunique:4
        column:OSVersion  nunique:27
        column:AntispamDirection  nunique:4
        column:SuspicionLevel  nunique:2
        column:LastVerdict  nunique:5
        column:CountryCode  nunique:162
        column:State  nunique:743
        column:City  nunique:2804
        column:IncidentGrade  nunique:3
        column:ActionGrouped  nunique:3
        column:ActionGranular  nunique:15

- It is very important to rebuild the chunks as first row of first chunk has the lowest timestamp there is i.e. timestamp is sorted acrss each chunk and throughout all chunks

- I did this the following way
```python
    for k in chunk_dict:
        chunk_dict[k] = (
            chunk_dict[k]
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )

    sorted_chunks = sorted(
        chunk_dict.items(),
        key=lambda x: x[1]["Timestamp"].iloc[0]
    )

    chunk_dict = {i: df for i, (_, df) in enumerate(sorted_chunks)}
```

This is the .info() for first chunk to get an idea:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 800000 entries, 0 to 799999
    Data columns (total 36 columns):
    #   Column             Non-Null Count   Dtype
    ---  ------             --------------   -----
    0   OrgId              800000 non-null  int64
    1   IncidentId         800000 non-null  int64
    2   Timestamp          800000 non-null  datetime64[ns, UTC]
    3   DetectorId         800000 non-null  int64
    4   AlertTitle         800000 non-null  int64
    5   Category           800000 non-null  object
    6   MitreTechniques    340473 non-null  object
    7   EntityType         800000 non-null  object
    8   EvidenceRole       800000 non-null  object
    9   Sha256             800000 non-null  int64
    10  IpAddress          800000 non-null  int64
    11  Url                800000 non-null  int64
    12  AccountUpn         800000 non-null  int64
    13  AccountName        800000 non-null  int64
    14  DeviceName         800000 non-null  int64
    15  EmailClusterId     8020 non-null    float64
    16  RegistryKey        800000 non-null  int64
    17  RegistryValueName  800000 non-null  int64
    18  RegistryValueData  800000 non-null  int64
    19  ApplicationName    800000 non-null  int64
    20  ThreatFamily       6203 non-null    object
    21  FileName           800000 non-null  int64
    22  FolderPath         800000 non-null  int64
    23  ResourceType       612 non-null     object
    24  Roles              18123 non-null   object
    25  OSFamily           800000 non-null  int64
    26  OSVersion          800000 non-null  int64
    27  AntispamDirection  14954 non-null   object
    28  SuspicionLevel     120822 non-null  object
    29  LastVerdict        187166 non-null  object
    30  CountryCode        800000 non-null  int64
    31  State              800000 non-null  int64
    32  City               800000 non-null  int64
    33  IncidentGrade      795641 non-null  object
    34  ActionGrouped      4761 non-null    object
    35  ActionGranular     4761 non-null    object
    dtypes: datetime64[ns, UTC] (1), float64(1), int64(21), object(13)
    memory usage: 219.7+ MB

## Test Data

 - *The test have almost 4.1 Million (i.e. 5 full chunks and last chunk with arounf 140000 rows) Rows of Data with 1 extra column as 'Usage' it has two kind of values 'Public', 'Private'*
 - It contains the labels as well we do not need to have a validation set
 we can use a chunk or two of the test set.