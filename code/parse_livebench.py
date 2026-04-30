"""
Parse LiveBench Table 3 (49 models × 6 categories) from the ICLR 2025 paper.
Source: White et al., LiveBench (arXiv:2406.19314), Table 3.
"""
import pandas as pd
import io

# Table 3 from the LiveBench paper, transcribed verbatim
# Columns: Model, LiveBench Score, Coding, Data Analysis, Instruction Following, Language, Math, Reasoning
table3_text = """\
claude-3-5-sonnet-20240620,61.2,63.2,56.7,72.3,56.9,53.7,64.0
gpt-4o-2024-05-13,55.0,46.4,52.4,72.2,53.9,49.9,55.0
gpt-4-turbo-2024-04-09,53.0,47.1,51.3,71.4,45.3,49.0,54.0
gpt-4-1106-preview,52.2,44.4,51.3,69.4,48.4,47.6,52.0
claude-3-opus-20240229,50.8,40.1,54.3,70.9,51.7,46.5,41.0
gpt-4-0125-preview,49.4,44.1,54.1,63.9,43.6,42.7,48.0
deepseek-coder-v2,46.8,41.1,38.3,67.2,33.0,52.2,49.0
gemini-1.5-pro-api-0514,44.4,32.8,52.8,67.2,38.3,42.1,33.0
gemini-1.5-flash-api-0514,40.9,39.1,44.0,63.0,30.7,38.5,30.0
qwen2-72b-instruct,40.2,31.8,26.2,68.3,29.2,43.4,42.0
mistral-large-2402,38.9,26.8,42.6,68.2,28.7,32.2,35.0
deepseek-chat-v2,38.4,33.5,38.0,64.3,32.3,33.2,29.0
claude-3-sonnet-20240229,38.1,25.2,44.6,65.0,38.1,29.6,26.0
meta-llama-3-70b-instruct,37.4,20.9,42.4,63.5,34.1,32.3,31.0
claude-3-haiku-20240307,35.3,24.5,41.5,64.0,30.1,25.7,26.0
mixtral-8x22b-instruct-v0.1,34.8,33.1,30.3,63.2,26.5,26.9,29.0
gpt-3.5-turbo-0125,34.4,29.2,41.2,60.5,24.2,25.5,26.0
gpt-3.5-turbo-1106,34.1,26.8,41.7,51.5,28.6,28.1,28.0
command-r-plus,32.9,20.3,24.6,71.5,23.9,24.9,32.0
mistral-small-2402,32.8,24.2,31.9,63.9,22.1,26.8,28.0
phi-3-medium-4k-instruct,30.3,20.6,31.6,53.3,13.9,27.5,35.0
phi-3-medium-128k-instruct,29.6,21.6,32.1,56.2,12.8,24.3,31.0
deepseek-coder-v2-lite-instruct,29.2,26.8,33.0,48.3,10.6,34.1,22.0
qwen1.5-110b-chat,29.0,22.2,31.5,55.3,13.2,25.6,26.0
qwen1.5-72b-chat,28.9,22.9,33.0,58.2,11.4,26.8,21.0
command-r,27.2,14.9,31.7,57.2,14.6,16.9,28.0
phi-3-small-128k-instruct,27.2,25.8,27.3,36.9,12.3,24.8,36.0
meta-llama-3-8b-instruct,26.7,18.3,23.3,57.1,18.7,17.6,25.0
qwen2-7b-instruct,26.5,29.2,28.7,44.7,10.2,25.8,20.0
phi-3-small-8k-instruct,26.2,19.6,27.5,48.2,15.0,24.1,23.0
openhermes-2.5-mistral-7b,23.3,11.6,26.9,52.8,11.4,20.1,17.0
mixtral-8x7b-instruct-v0.1,22.5,11.3,28.1,44.8,13.8,19.0,18.0
mistral-7b-instruct-v0.2,19.3,11.6,14.6,51.6,9.1,16.0,13.0
phi-3-mini-4k-instruct,19.3,14.9,14.7,40.1,7.1,19.9,19.0
zephyr-7b-alpha,19.2,11.3,17.4,52.8,7.2,9.6,17.0
phi-3-mini-128k-instruct,18.0,11.6,8.7,49.6,6.8,21.5,10.0
zephyr-7b-beta,17.3,8.3,15.7,48.3,4.3,11.2,16.0
deepseek-v2-lite-chat,17.1,8.6,18.2,41.8,9.2,12.0,13.0
qwen1.5-7b-chat,16.5,6.6,16.2,44.1,6.2,12.9,13.0
starling-lm-7b-beta,16.4,18.3,2.0,38.3,7.3,13.8,19.0
vicuna-7b-v1.5-16k,13.7,1.3,9.3,42.1,7.9,6.6,15.0
vicuna-7b-v1.5,11.7,1.0,2.7,41.8,8.7,4.3,12.0
qwen1.5-4b-chat,11.1,4.0,9.1,27.7,5.8,7.1,13.0
llama-2-7b-chat-hf,10.3,0.0,0.0,44.9,6.9,4.8,5.0
qwen2-1.5b-instruct,10.0,5.6,10.0,25.9,3.0,7.2,8.0
yi-6b-chat,8.8,1.3,4.4,27.2,4.7,7.1,8.0
qwen2-0.5b-instruct,6.8,2.0,2.0,26.6,2.8,4.2,3.0
qwen1.5-1.8b-chat,6.1,0.0,3.3,22.9,3.2,2.1,5.0
qwen1.5-0.5b-chat,5.3,0.0,0.0,21.3,2.9,3.4,4.0
"""

cols = ['model', 'overall', 'coding', 'data_analysis', 'instruction_following',
        'language', 'math', 'reasoning']
df = pd.read_csv(io.StringIO(table3_text), names=cols, header=None)

print(f"N models: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nHead:\n{df.head()}")
print(f"\nTail:\n{df.tail()}")
print(f"\nDescribe:\n{df.describe().round(2)}")

# Convert to [0,1] scale to match HF
score_cols = ['coding', 'data_analysis', 'instruction_following', 'language', 'math', 'reasoning']
df_scaled = df.copy()
for c in score_cols:
    df_scaled[c] = df_scaled[c] / 100.0
df_scaled['overall'] = df_scaled['overall'] / 100.0

df_scaled.to_csv('../data/livebench_scores.csv', index=False)
print(f"\nSaved {len(df_scaled)} models × 6 categories to livebench_scores.csv")
print(f"\nSanity check: row means should equal 'overall':")
df_scaled['computed_mean'] = df_scaled[score_cols].mean(axis=1)
print(df_scaled[['model', 'overall', 'computed_mean']].head())
