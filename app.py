import gradio as gr
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from configuration import Config
from onnx_model import ONNXModel
from postprocess import get_sentiment
from theme import theme
from utils import download_model

hf_logging.disable_progress_bar()
config = Config()

model_path = download_model("ml-sentiment-adapter", "production")
model = ONNXModel.from_dir(model_path)
tokenizer = AutoTokenizer.from_pretrained(model.model_info.base_model)


def predict(sentence: str):
    encoding = tokenizer([sentence], truncation=True, return_tensors="np")
    logits = model(**encoding)
    score, sentiment = get_sentiment(logits, config.negative_threshold, config.positive_threshold, config.zero)
    result = {
        sentiment: score
    }
    return result


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Customer Review", value="Lettria truly handled all the overhead of an NLP project!"),
    outputs=gr.Label(label="Sentiment Level"),
    title="Lettria's Customer Sentiment Analysis",
    description="Introducing our Sentiment Analysis API powered by Deep Learning! It provides an easy-to-use solution for analyzing and understanding the sentiment expressed in text. With this API, you can gain valuable insights from customer feedback and reviews by accurately classifying text into positive, negative, or neutral sentiment categories. Seamlessly integrate it into your applications to make data-driven decisions, monitor brand reputation, and enhance customer satisfaction in real-time. Uncover the true sentiment behind text and unlock the power of sentiment analysis today!",
    examples=[
        "I absolutely loved the movie! The storyline was captivating, and the acting was superb.",
        "I'm extremely disappointed with the quality of the product. It broke within a week of use.",
        "Today has been an average day. Nothing particularly good or bad happened.",
        "This book is a masterpiece. The author's writing style is brilliant, and the characters are well-developed.",
        "I'm feeling neutral about the new restaurant. The ambiance was nice, but the food was mediocre.",
    ],
    theme=theme,
    allow_flagging="never",
)

demo.launch()# sentiment-analysis-on-customer-s-feedback-on-products
