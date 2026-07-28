# Dataset Description 


### First Impressions from the Data

 - The Training data has 9,516,837 rows
 - Since I have to do this on Google Colab because of unavailability of GPU and limited RAM and storage the data still seems be crashing the RAM given by colab
 - To solve this we come up with chunks of data each consisting a set of 800000 rows chunk size which gives us 12 chunks last chunk (chunk 11) is of size 716837 rows
 ```python
chunks = pd.read_csv(
    "drive/MyDrive/GUIDE_Train.csv",
    chunksize=800_000
)
```
 - We need to make a dictionary chunk_dict={} to get full data as follows 
```python
chunk_dict = {i: chunk for i, chunk in enumerate(chunks)}
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
  - **DeviceId** – Device-specific ID
  - **AccountSid** – User SID
  - **AccountObjectId** – Azure object ID
  - **NetworkMessageId** – Message ID
  - **ApplicationId** – Identifier only
  - **OAuthApplicationId** – Identifier only
  - **ResourceIdName** – Mostly unique identifier

Hence,
```python
input_cols = ['OrgId', 'IncidentId', 'Timestamp', 'DetectorId', 'AlertTitle', 
              'Category', 'MitreTechniques', 'EntityType', 'EvidenceRole', 'Sha256',
              'IpAddress', 'Url', 'AccountUpn', 'AccountName', 'DeviceName', 
              'EmailClusterId', 'RegistryKey', 'RegistryValueName', 
              'RegistryValueData', 'ApplicationName', 'ThreatFamily', 'FileName',
              'FolderPath', 'ResourceType', 'Roles', 'OSFamily', 'OSVersion', 
              'AntispamDirection', 'SuspicionLevel', 'LastVerdict', 'CountryCode',
              'State', 'City']
```

and

```python
target_cols=['IncidentGrade','ActionGrouped','ActionGranular']
```

#### Target Features

 - IncidentGrade : The final triage label assigned to an alert, indicating whether it is a True Positive, Benign Positive, or False Positive.

 - ActionGrouped : The high-level remediation action recommended for handling the incident, such as Investigate, Block, or Isolate Device.

 - ActionGranular – The specific remediation step to take for the incident, providing a more detailed action than ActionGrouped (e.g., Block File Hash, Disable User Account, Kill Process).

#### Data Profiling

 - 