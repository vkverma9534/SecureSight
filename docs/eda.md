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
 'OSFamily','OSVersion','CountryCode','State','City','DeviceId','ApplicationId',]

 - Categorical columns = ['Category','MitreTechniques','EntityType','EvidenceRole',
 'ThreatFamily','ResourceType','Roles','AntispamDirection','SuspicionLevel',
 'LastVerdict','IncidentGrade','ActionGrouped',ActionGranular','ResourceIdName']

 - Datetime columns = ['Timestamp']
 - Notice that columns like City Countrycode etc are categorical but we can not hot encode them since they have many values (high cardinality) and will result in unecessary expansion of data
 also Label encoding (The form that data is already in) is very misleading to the models since they are categorical they might trigger neighbour values and throw ambiguous models.

 - This requires us to redefine what true numeric cols are rather than just choosing select_dtypes

 - Feature wise EDA
    - 'OrgId'
        - There is no need of standardisation since no nulls and no data type othere than int64 in any chunk
```python
              for i in range(len(chunk_dict)):
                     s = chunk_dict[i]['OrgId']
                     non_numeric = s[~s.apply(lambda x: isinstance(x, (int, float)))]
                     print(non_numeric.unique())
```
        - No Missing Value
        - Interestingly it does not have that high cardinality(with respect to size of data) but the cardinality is variable in chunks so some models might have no idea for some values of OrgId although this variation is not very significant
        4279,4332,4283,4285,4323,4307,4262,4320,4294,4167,4291,4293 
        in total taking union of all there are 5769 (Actual cardinality)unique 'OrgId' in training data
```python
             for i in range(len(chunk_dict)):
                 print(chunk_dict[i]['OrgId'].nunique())
```
        - Frequency Distibution
```python
              freq = (
                     pd.concat([chunk_dict[i]['OrgId'] for i in chunk_dict])
                     .value_counts(dropna=False)
                     .rename('count')
                     .reset_index()
              )

              freq.columns = ['OrgId', 'count']
              freq['count_pct'] = ((freq['count'] / total_valid_rows) * 100).round(2).astype(str) + '%'
```
              - Top 20 Orgs
                     OrgId   count count_pct
                     0       0  844789      8.9%
                     1       2  228325     2.41%
                     2       1  210044     2.21%
                     3       3  190866     2.01%
                     4       5  173431     1.83%
                     5       6  161092      1.7%
                     6       4  145741     1.54%
                     7       7  134532     1.42%
                     8       8  133637     1.41%
                     9      10  133160      1.4%
                     10      9  130807     1.38%
                     11     11  116134     1.22%
                     12     12  114799     1.21%
                     13     14  112681     1.19%
                     14     13  107259     1.13%
                     15     16   87836     0.93%
                     16     25   83539     0.88%
                     17     17   81424     0.86%
                     18     19   80168     0.84%
                     19     18   78881     0.83%
              - Tail of Data: 1845 categories have their appearence less than 10 times (1123 of them are less than 5 times)
              - Interesting fact that only around top 25 categories are 40% volume of the data (top 50 are more than 50) this is interesting as there are 5769 total categories
              - By frequency distribution 
                   appearance  participation
              0      10000+            156
              1  1000-10000            600
              2    100-1000           1233
              3      10-100           1935
              4        5-10            722
              5         3-5            911
              6         1-3            212
              ![Frequency Distribution](images/freq_distri_org_id.png)