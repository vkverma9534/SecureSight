## Ideas
 - I do not think removing data helps instead we should model some data that work on non null data while some that do not need such data. Since some columns seems to be having no nulls anywhere throughout the dataset so we are assumin them to be non nulls in test.

 - We should make the data indepependent of timestamp but to preserve the Seasonal nature we can divide timestamp into season, month, date_of_month, week_of_month, day_of_week and time_of_day.
- Catboost Encoding is a good to go way also Target encoding can be useful other than these many models have their native categorical handling so we can rely on that as well(like: XGBoost Classifier, CatBoost Classifier, LightGBM)
- For High Cardinality data having a long tail (many categories appear less than 10 or even less than 5 times) we can combine it a a different section (say other)
- Make a entropy for each unique for a feature to help model analyse more better



## Actions
    - Made chunks of data
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
    - Removed columns that are no contributing in analysis(with reasons) from test as well as validation set
        - **Id** – Pure row identifier
        - **AlertId** – Unique alert identifier
        - **AccountSid** – User SID
        - **AccountObjectId** – Azure object ID
        - **NetworkMessageId** – Message ID
        - **OAuthApplicationId** – Identifier only


    - Sorted timestamp across chunks and inside chunk
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
    - Created training set and validation set (by choosing the first chunk of data from test_set)
```python
        for i in range (12):
            chunk_dict[i]=chunk_dict[i][input_cols+target_cols]
        val_df=test_chunk_dict[0][input_cols+target_cols]
```
    - Dropping Duplicate rows from all chunks (Final volume for chunk 797714,797743,797642,797670,797730,797682,797676,797694,797685,714985,797662,797594)
```python
        for i in range(len(chunk_dict)):
            chunk_dict[i]=chunk_dict[i].drop_duplicates()
            print(len(chunk_dict[i]))
```  
    - 