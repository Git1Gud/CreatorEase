import pandas as pd

df1 = pd.read_csv('youtube_shorts_podcast_dataset_cleaned.csv')
df2 = pd.read_csv('youtube_shorts_podcast_dataset_with_qa.csv')

# Merge question_text and answer_text from df1 into df2 based on video_id
df2 = df2.merge(
    df1[['video_id', 'question_text', 'answer_text']],
    on='video_id',
    how='left',
    suffixes=('', '_from_df1')
)

# Update only where video_id matches
df2['question_text'] = df2['question_text_from_df1']
df2['answer_text'] = df2['answer_text_from_df1']

# Drop the extra columns
df2 = df2.drop(columns=['question_text_from_df1', 'answer_text_from_df1'])

# Save the updated df2 if needed
df2.to_csv('youtube_shorts_podcast_dataset_with_qa1.csv', index=False)