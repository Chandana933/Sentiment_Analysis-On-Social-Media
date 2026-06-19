import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import os

# Load Dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "sample_data.csv")

df = pd.read_csv(csv_path)

# Sentiment Function
def get_sentiment(text):
    analysis = TextBlob(text)

    if analysis.sentiment.polarity > 0.1:
        return "Positive"
    elif analysis.sentiment.polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Apply Sentiment Analysis
df["Sentiment"] = df["Post"].apply(get_sentiment)

# Calculate Polarity
df["Polarity"] = df["Post"].apply(
    lambda x: TextBlob(x).sentiment.polarity
)

# Display Results
print("\nSentiment Analysis Results\n")
print(df[["User", "Post", "Sentiment", "Polarity", "Likes"]])

# Sentiment Counts
sentiment_counts = df["Sentiment"].value_counts()

# Create Output Folder
os.makedirs("../outputs", exist_ok=True)

# Visualization 1
plt.figure(figsize=(8,5))
sentiment_counts.plot(kind='bar')

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("../outputs/sentiment_distribution.png")
plt.show()

# Visualization 2
plt.figure(figsize=(8,5))

avg_likes = df.groupby("Sentiment")["Likes"].mean()

avg_likes.plot(kind='bar')

plt.title("Average Likes by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Average Likes")

plt.tight_layout()
plt.savefig("../outputs/average_likes.png")
plt.show()

# Save Results
df.to_csv("../outputs/social_media_sentiment_results.csv",
          index=False)

print("\nAnalysis Completed Successfully!")
print("Results saved in outputs folder.")