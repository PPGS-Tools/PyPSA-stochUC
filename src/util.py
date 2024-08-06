import pandas as pd
from pandas import DataFrame

import os
current = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR =  os.path.dirname(current)

def testData()->DataFrame:
    """Loads the historic sample data included in this project.

    This data is in hourly resolution and partly consists of artificial ("historic") data.
    In particular, missing datapoints were filled with artificial values like mean of neighbours.
    The district heat demand is completely artificial, based on a trained model.

    Returns
    -------
    DataFrame
        Historic data in hourly resolution, without NaNs

    Examples
    -------
    >>> df = testData()
    >>> df
                         aFRR_neg_EUR_MW  aFRR_pos_EUR_MW    temp_C  residual_load_MW  da_EUR_MWh  q_fern_MW
    time_UTC                                                                                                
    2019-01-01 00:00:00           36.605             7.35  1.933333           17286.0       10.07  14.865311
    2019-01-01 01:00:00           36.605             7.35  1.850000           14071.0       -4.08  14.694157
    2019-01-01 02:00:00           36.605             7.35  1.650000           11795.0       -9.91  14.958593
    2019-01-01 03:00:00           36.605             7.35  1.500000           10568.0       -7.41  15.595638
    2019-01-01 04:00:00           40.190             5.80  1.650000           10087.0      -12.55  16.444387
    ...                              ...              ...       ...               ...         ...        ...
    2021-12-31 19:00:00            4.380             9.35  6.066667           12987.0        0.18  17.602586
    2021-12-31 20:00:00            8.410             2.13  6.350000           11225.0        0.08  17.138763
    2021-12-31 21:00:00            8.410             2.13  6.516667           11460.0        5.10  16.426286
    2021-12-31 22:00:00            8.410             2.13  6.433333           10640.0        6.32  15.615477
    2021-12-31 23:00:00            8.410             2.13  6.333333           11000.0       50.05  14.854040
    """
    pathTestData = os.path.join(ROOT_DIR,"data/historic_data")
    dfs: list[DataFrame] = []
    for dirEntry in os.scandir(pathTestData):
        if dirEntry.is_file():
            inputFile = dirEntry
            df = pd.read_csv(inputFile,index_col=0)
            df.index = pd.DatetimeIndex(df.index,freq="infer")
            dfs.append(df)
    df = pd.concat(dfs,axis=1)
    df["Weekend"] = df.index.map(lambda x: int(x.day_of_week > 4))        
    return df

if __name__ == "__main__":
    # df = testData()
    # print(df)
    import doctest
    doctest.testmod()