import boto3

s3 = boto3.client("s3")
s3.download_file("fake-news-models-files", "nb_fake_news_model.joblib", "nb_fake_news_model.joblib")