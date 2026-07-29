# Exploratory Data Analysis

 - Total number of rows = 9516837
 - Columns not impotant for analysis
        - **Id** – Pure row identifier
        - **AlertId** – Unique alert identifier
        - **AccountSid** – User SID
        - **AccountObjectId** – Azure object ID
        - **NetworkMessageId** – Message ID
        - **OAuthApplicationId** – Identifier onlyidentifier
 - Target Columns - 'IncidentGrade','ActionGrouped','ActionGranular' (All are Object type)
 - Input Columns - 'OrgId', 'IncidentId', 'Timestamp', 'DetectorId', 'AlertTitle',
              'Category', 'MitreTechniques', 'EntityType', 'EvidenceRole', 'Sha256',
              'IpAddress', 'Url', 'AccountUpn', 'AccountName', 'DeviceName',
              'EmailClusterId', 'RegistryKey', 'RegistryValueName',
              'RegistryValueData', 'ApplicationName', 'ThreatFamily', 'FileName',
              'FolderPath', 'ResourceType', 'Roles', 'OSFamily', 'OSVersion',
              'AntispamDirection', 'SuspicionLevel', 'LastVerdict', 'CountryCode',
              'State', 'City','DeviceId','ApplicationId','ResourceIdName'
```python
            numeric_cols=chunk_dict[0].select_dtypes(include=np.number).columns.tolist()
            categorical_cols=chunk_dict[0].select_dtypes(include=object).columns.tolist()
            datetime_cols = chunk_dict[0].select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
```
 - Numeric columns(*Remember This is just the Initial Speculation as many or all of these are categorical cols with label encoding, each feature will be analyzed seperately*) =
  ['OrgId','IncidentId','DetectorId','AlertTitle','Sha256',
 'IpAddress','Url','AccountUpn','AccountName''DeviceName','EmailClusterId',
 'RegistryKey','RegistryValueName','RegistryValueData','ApplicationName','FileName','FolderPath',
 'OSFamily','OSVersion','CountryCode','State','City']

 - Categorical columns = ['Category','MitreTechniques','EntityType','EvidenceRole',
 'ThreatFamily','ResourceType','Roles','AntispamDirection','SuspicionLevel',
 'LastVerdict','IncidentGrade','ActionGrouped',ActionGranular']

 - Datetime columns = ['Timestamp']
 - Notice that columns like City Countrycode etc are categorical but we can not hot encode them since they have many values (high cardinality) and will result in unecessary expansion of data
 also Label encoding (The form that data is already in) is very misleading to the models since they are categorical they might trigger neighbour values and throw ambiguous models.

 - This requires us to redefine what true numeric cols are rather than just choosing select_dtypes