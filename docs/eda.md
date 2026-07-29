# Exploratory Data Analysis

 - Total number of rows = 9516837
 - Columns not impotant for analysis
        - **Id** – Pure row identifier
        - **AlertId** – Unique alert identifier
        - **DeviceId** – Device-specific ID
        - **AccountSid** – User SID
        - **AccountObjectId** – Azure object ID
        - **NetworkMessageId** – Message ID
        - **ApplicationId** – Identifier only
        - **OAuthApplicationId** – Identifier only
        - **ResourceIdName** – Mostly unique identifier
 - Target Columns - 'IncidentGrade','ActionGrouped','ActionGranular' (All are Object type)
 - Input Columns - 'OrgId', 'IncidentId', 'Timestamp', 'DetectorId', 'AlertTitle',
              'Category', 'MitreTechniques', 'EntityType', 'EvidenceRole', 'Sha256',
              'IpAddress', 'Url', 'AccountUpn', 'AccountName', 'DeviceName',
              'EmailClusterId', 'RegistryKey', 'RegistryValueName',
              'RegistryValueData', 'ApplicationName', 'ThreatFamily', 'FileName',
              'FolderPath', 'ResourceType', 'Roles', 'OSFamily', 'OSVersion',
              'AntispamDirection', 'SuspicionLevel', 'LastVerdict', 'CountryCode',
              'State', 'City'
```python
            numeric_cols=chunk_dict[0].select_dtypes(include=np.number).columns.tolist()
            categorical_cols=chunk_dict[0].select_dtypes(include=object).columns.tolist()
            datetime_cols = chunk_dict[0].select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
```
 - Numeric columns = ['OrgId','IncidentId','DetectorId','AlertTitle','Sha256',
 'IpAddress','Url','AccountUpn','AccountName''DeviceName','EmailClusterId',
 'RegistryKey','RegistryValueName','RegistryValueData','ApplicationName','FileName','FolderPath',
 'OSFamily','OSVersion','CountryCode','State','City']

 - Categorical columns = ['Category','MitreTechniques','EntityType','EvidenceRole',
 'ThreatFamily','ResourceType','Roles','AntispamDirection','SuspicionLevel',
 'LastVerdict','IncidentGrade','ActionGrouped',ActionGranular']

 - Datetime columns = ['Timestamp']
 - Notice that columns like City Countrycode etc are categorical but we can not hot encode them since they have many values (high cardinality) and will result in unecessary expansion of data

 - This requires us to redefine what true numeric cols are rather than just choosing select_dtypes