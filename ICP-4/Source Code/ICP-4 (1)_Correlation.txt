import pandas as pd
train_df= pd.read_csv('./train.csv')
test_df= pd.read_csv('./test.csv')
combine = [train_df, test_df]

print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))