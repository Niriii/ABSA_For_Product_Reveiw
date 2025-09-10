# **Aspect-Based Sentiment Analysis (ABSA) Project**





#### **📌 Project Objective**



The goal of this project is to build a system that can automatically identify specific aspects (e.g., battery life, screen quality, price) mentioned in product reviews and assign a sentiment (Positive, Negative, Neutral) to each aspect.



This helps businesses and analysts understand customer opinions at a more granular level rather than just overall sentiment.



#### **📂 Project Structure**



project\_root/

│

├── flipkart\_reviews\_dataset.csv     # Raw dataset containing product reviews

├── analysis.ipynb                   # Jupyter notebook for analysis \& structured outputs

├── aspect\_sentiment\_pipeline.py     # Pipeline function for aspect extraction \& sentiment classification

├── README.txt                       # Project documentation



#### **⚙️ Requirements**



Make sure you have Python 3.9+ installed.



Install dependencies:

pip install pandas transformers pyabsa



Additional requirements (if not already installed):

pip install torch --index-url https://download.pytorch.org/whl/cu129

(adjust the PyTorch version depending on your CUDA/CPU setup)



#### **🚀 How to Run the Project**



##### 1\. Explore in Jupyter Notebook



Open analysis.ipynb to:



* Perform preprocessing and cleaning.
* Run aspect extraction and sentiment classification.
* extract structured outputs.



##### 2\. Run the Pipeline Script

You can use the provided pipeline function directly:



code :



from aspect\_sentiment\_pipeline import aspect\_sentiment\_pipeline



\# Example: Running on CSV dataset

results = aspect\_sentiment\_pipeline("flipkart\_reviews\_dataset.csv", csv\_column="review\_text")



\# Example: Running on a list of reviews

reviews = \[

    "The battery life is amazing but the screen quality is poor.",

    "Great price, but the delivery was late."

]

results = aspect\_sentiment\_pipeline(reviews)



The structured output will be saved to:



aspect\_sentiment\_results.json



#### **💡Important Note**



* The pipeline only extracts aspects and assigns sentiments.
* It does not perform dataset cleaning.



For better results, it is strongly recommended to provide a cleaned dataset (remove noise, convert to lowercase, etc.) before passing it to the pipeline.



#### **📊 Output Format**



Each review is transformed into structured JSON with review text, aspects, and sentiments:



{

  "review": "The battery life is amazing but the screen quality is poor.",

  "aspects": \["battery life", "screen quality"],

  "sentiments": \["Positive", "Negative"]

}



#### **💡 Notes**



* The pipeline is GPU-optimized (CUDA support) but will also run on CPU.
* If the pipeline is running on CPU, it is recommended to update the code with batch processing so that multiple reviews can be processed efficiently without slowing down performance or consuming excessive memory.
* Long reviews are automatically split into chunks for better performance.
* If no aspects are found, the system assigns an overall product sentiment.



#### **🔮 Future Improvements**



* Add support for multilingual reviews.
* Improve aspect phrase grouping (e.g., "display" and "screen" → same aspect).
* Build an interactive dashboard for real-time review insights.

