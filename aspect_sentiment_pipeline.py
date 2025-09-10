#%pip install pyabsa
import pandas as pd
import os
import re
import json
from transformers import pipeline, AutoTokenizer
from pyabsa import AspectTermExtraction as ATEPC


def aspect_sentiment_pipeline(
    input_data,
    csv_column='review_text',
    output_file='aspect_sentiment_results.json',
    max_chunk_length=256
):
    """
    Perform aspect extraction and sentiment classification on reviews from a CSV file or list.
    Optimized for GPU (CUDA): removes manual batch loop but keeps review chunking.
    """

    # ---- PATCH: Load tokenizer and add missing bos/eos tokens ----
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "[CLS]"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "[SEP]"})
    # --------------------------------------------------------------

    # Load aspect extractor with CUDA
    aspect_extractor = ATEPC.AspectExtractor(
        "english",
        auto_device=True,   # will auto-select GPU if available
        tokenizer=tokenizer,
        sentiment_model="cardiffnlp/twitter-roberta-base-sentiment"
    )

    # Standalone fallback sentiment classifier (CUDA enabled)
    sentiment_classifier = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=0  # force GPU if available
    )

    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }

    # Handle input: CSV or list
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"File '{input_data}' does not exist")
        df = pd.read_csv(input_data)
        if csv_column not in df.columns:
            raise ValueError(f"CSV file does not contain '{csv_column}' column. Available: {list(df.columns)}")
        reviews = df[csv_column].astype(str).tolist()
    else:
        reviews = [str(r) for r in input_data if isinstance(r, str)]
        if not reviews:
            raise ValueError("No valid string reviews provided.")

    # ---- Preprocess reviews (keep chunks for long reviews) ----
    def split_review(review):
        review = re.sub(r'[^\x00-\x7F]+', ' ', review)
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', review)
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks if chunks else [review]

    processed_reviews, original_to_chunks_map = [], []
    for review in reviews:
        chunks = split_review(review) if len(review) > max_chunk_length * 2 else [review]
        processed_reviews.extend(chunks)
        original_to_chunks_map.append((len(chunks), review))

    # ---- GPU-parallel aspect extraction (no batch loop) ----
    aspect_results = aspect_extractor.predict(
        processed_reviews,
        print_result=False,
        pred_sentiment=True
    )

    # ---- Aggregate results per original review ----
    final_results, chunk_index, empty_aspect_count = [], 0, 0
    for num_chunks, original_review in original_to_chunks_map:
        aspect_to_sentiment = {}
        for _ in range(num_chunks):
            if chunk_index < len(aspect_results):
                chunk_result = aspect_results[chunk_index]
                aspects = chunk_result.get('aspect', [])
                sentiments = chunk_result.get('sentiment', [])
                if not aspects:
                    empty_aspect_count += 1
                for a, s in zip(aspects, sentiments):
                    aspect_to_sentiment[a] = label_map.get(s, s)
                chunk_index += 1

        aggregated_aspects = list(aspect_to_sentiment.keys())
        aggregated_sentiments = [aspect_to_sentiment[a] for a in aggregated_aspects]

        # Fallback: no aspects found → classify overall sentiment
        if not aggregated_aspects:
            aggregated_aspects = ["overall product"]
            try:
                result = sentiment_classifier(original_review, text_pair="overall product")
                aggregated_sentiments = [label_map.get(result[0]['label'], "Neutral")]
            except Exception:
                aggregated_sentiments = ["Neutral"]
            empty_aspect_count += 1

        final_results.append({
            "review": original_review,
            "aspects": aggregated_aspects,
            "sentiments": aggregated_sentiments
        })

    # ---- Save results ----
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    return final_results