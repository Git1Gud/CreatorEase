import pandas as pd

df1 = pd.read_csv('youtube_shorts_podcast_dataset_with_qa.csv')
df2 = pd.read_csv('youtube_shorts_podcast_dataset_cleaned.csv')

# Merge on video_id, keeping only columns from df1 and updating with df2 where available
df3 = df1.copy()
df3 = df3.merge(
    df2.set_index('video_id')[['question_text', 'answer_text']],
    left_on='video_id',
    right_index=True,
    how='left',
    suffixes=('', '_from_df2')
)

# Update question_text and answer_text only where available in df2
df3['question_text'] = df3['question_text_from_df2'].combine_first(df3['question_text'])
df3['answer_text'] = df3['answer_text_from_df2'].combine_first(df3['answer_text'])

df3 = df3[df1.columns]

print(df3.head())

df3.to_csv('youtube_shorts_podcast_dataset_with_qa2.csv', index=False)


