for i in numeric_cols
    for i in range(len(chunk_dict)):
        s = chunk_dict[i]['i']
        non_numeric = s[~s.apply(lambda x: isinstance(x, (int, float)))]
        print(non_numeric.unique())
    for i in range(len(chunk_dict)):
        freq=chunk_dict[i]['i'].value_counts(dropna=False)
        print(freq.head(20))
        print(chunk_dict[i]['i'].nunique())
    freq=freq.reset_index()
    print("freq")
    print(freq)
    freq = (
        pd.concat([chunk_dict[i]['i'] for i in chunk_dict])
          .value_counts(dropna=False)
          .rename('count')
          .reset_index()
    )
    freq=freq.sort_values('count',ascending=False)
    freq['count_pct(%)']=((freq['count']/9516837)*100).round(3)
    freq['cumsum_pct']=freq['count_pct(%)'].cumsum()
    freq.columns = ['i', 'count','count_pct(%)','cumsum_pct']
    print('freq.head(50)')
    print(f"freq_size:{len(freq)}")
    print(freq.head(50))
    total_valid_rows=0
    for i in range(len(chunk_dict)):
        total_valid_rows+=len(chunk_dict[i])
    freq['count_pct'] = ((freq['count'] / total_valid_rows) * 100).round(2)
    print("freq[freq['count']<3]")
    print(freq[freq['count']<3])
    freq['cumsum_count'] = (((freq['count'] / total_valid_rows) * 100).round(2)).cumsum()
    len(freq[(10000>freq['count']) & (freq['count']>1000)])
    freq_distri = pd.DataFrame({
        'appearance': ['10000+', '1000-10000', '100-1000', '10-100', '5-10', '3-5', '1-3'],
        'participation': [
            len(freq[freq['count'] >= 10000]),
            len(freq[(freq['count'] >= 1000) & (freq['count'] < 10000)]),
            len(freq[(freq['count'] >= 100) & (freq['count'] < 1000)]),
            len(freq[(freq['count'] >= 10) & (freq['count'] < 100)]),
            len(freq[(freq['count'] >= 5) & (freq['count'] < 10)]),
            len(freq[(freq['count'] >= 3) & (freq['count'] < 5)]),
            len(freq[(freq['count'] >= 1) & (freq['count'] < 3)])
        ]
    })
    import plotly.express as px

    fig = px.bar(
        freq_distri,
        x='participation',
        y='appearance',
        orientation='h',
        text='participation'
    )

    # fig.update_traces(
    #     textposition='outside',
    #     texttemplate='%{text}'
    # )

    fig.update_layout(template='plotly_white')

    fig.show()
    df=freq.copy()
    df=df.head(50)
    df=df.sort_values('count_pct')
    print(df.info())
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Rank"],
            y=df["count_pct"],
            mode="lines",
            line=dict(width=2)
        )
    )

    fig.update_layout(
        template="plotly_white",
        title="Long-tail Distribution of i",
        xaxis_title="i Rank (sorted by frequency)",
        yaxis_title="Percentage of Records"
    )

    fig.show()
    print('df.shape')
    print(df.shape)
    print('df.head()')
    print(df.head())
    print('df.dtypes')
    print(df.dtypes)
    print("df[['i', 'count_pct']].head(10)")
    print(df[['i', 'count_pct']].head(10))
    dfs = []

    for i in range(len(chunk_dict)):
        df = chunk_dict[i][['i', 'IncidentGrade']].copy()

        category_counts = (
            df.groupby(['i', 'IncidentGrade'])
              .size()
              .unstack(fill_value=0)
        )

        dfs.append(category_counts)

    result = (
        pd.concat(dfs)
          .groupby('i')
          .sum()
          .reset_index()
    )

    result.columns.name = None
    result['total_count']=result['FalsePositive']+result['BenignPositive']+result['TruePositive']
    result['FalsePositive_pct']=((result['FalsePositive']/result['total_count'])*100).round(3)
    result['BenignPositive_pct']=((result['BenignPositive']/result['total_count'])*100).round(3)
    result['TruePositive_pct']=((result['TruePositive']/result['total_count'])*100).round(3)
    print('result')
    print(result)
    df=result[['i','total_count','FalsePositive_pct','BenignPositive_pct','TruePositive_pct']][result['total_count']>10000]
    BenignPositive_df=df.sort_values(by=['BenignPositive_pct','total_count'],ascending=[False,False])
    print('BenignPositive_df.head(20)')
    print(BenignPositive_df.head(20))
    FalsePositive_df=df.sort_values(by=['FalsePositive_pct','total_count'],ascending=[False,False])
    print('FalsePositive_df.head(20)')
    print(FalsePositive_df.head(20))
    TruePositive_df=df.sort_values(by=['TruePositive_pct','total_count'],ascending=[False,False])
    print('TruePositive_df.head(20)')
    print(TruePositive_df.head(20))
    import matplotlib.pyplot as plt
    target_cols = ['FalsePositive', 'BenignPositive', 'TruePositive']
    result['max_target_pct'] = result[target_cols].max(axis=1) / result['total_count'] * 100
    result['max_target_pct'].describe()
    plt.figure(figsize=(8,5))
    plt.hist(result['max_target_pct'], bins=30)
    plt.xlabel("Dominant Class Percentage")
    plt.ylabel("Number of i")
    plt.title("Distribution of Dominant Target Percentage")
    plt.show()
