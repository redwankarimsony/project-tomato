import pandas as pd

def process(df_file="train.csv"):
    df = pd.read_csv(df_file)
    print(df.head())
    for i in range(df.shape[0]):
        df.iloc[i, 0]= df.iloc[i, 0].replace("/Tomato/", "/")
    print(df.head())

    df.to_csv(df_file, index=True)



if __name__ =="__main__":
    process("train.csv")
    process("test.csv")
    process("valid.csv")