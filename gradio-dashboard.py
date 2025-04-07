import pandas as pd
import numpy as np
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from scripts.regsetup import description

load_dotenv()
books=pd.read_csv("books_with_emotions.csv")
books["large thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large thumbnail"] = np.where(books["thumbnail"].isna(),"cover-not-found.png",books["large thumbnail"])

raw_documents=TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter=CharacterTextSplitter(chunk_size=0,chunk_overlap=0,separator="\n")
documents=text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents,
                                embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
                                collection_name="book_collection")

def retrieve_semantic_recommendations(
        query:str,
        category:str = None,
        tone:str = None,
        initial_top_k:int=50,
        final_top_k:int=16,
)->pd.DataFrame:

    recs=db_books.similarity_search(query,k=initial_top_k)
    books_list=[int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_rec=books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_rec=books[books["simple_categories"] == category].head(final_top_k)
    else:
        book_rec=book_rec.head(final_top_k)
    if tone=="Happy":
        book_rec.sort_values(by="joy",ascending=False,inplace=True)
    elif tone=="Surprising":
        book_rec.sort_values(by="surprise",ascending=True,inplace=True)
    elif tone=="Angry":
        book_rec.sort_values(by="anger",ascending=False,inplace=True)
    elif tone=="Suspenseful":
        book_rec.sort_values(by="fear",ascending=True,inplace=True)
    elif tone=="Sad":
        book_rec.sort_values(by="sadness",ascending=False,inplace=True)

    return book_rec

def recommend_books(
        query:str,
        category:str,
        tone:str
):
    recommendations=retrieve_semantic_recommendations(query,category,tone)
    results=[]
    for _, row in recommendations.iterrows():
        description=row["description"]
        truncated_desc_split=description.split()
        truncated_description=" ".join(truncated_desc_split[:30]) + "..."
        authors_split=row["authors"].split(";")
        if len(authors_split)==2:
            authors_str=f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split)>2:
            authors_str=f"{' , '.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors_str=row["authors"]
        caption=f"{row["title"]} by {authors_str}: {truncated_description}"
        results.append((row["large thumbnail"],caption))
    return results

categories=["All"] + sorted(books["simple_categories"].unique())
tones=["All"] + ["Happy","Surprising","Angry","Suspenseful","Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")
    with gr.Row():
        user_query= gr.Textbox(label = "Please enter a description of a book",
                               placeholder="e.g. , A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories,label="Select a Category:",value = "All")
        tone_dropdown = gr.Dropdown(choices = tones,label="Select a Tone:",value = "All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books",columns=8,rows=2)
    submit_button.click(fn=recommend_books,
                        inputs=[user_query,category_dropdown,tone_dropdown] ,
                        outputs=[output])

    if __name__ == "__main__":
        dashboard.launch()
