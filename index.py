import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def main():
    st.set_page_config(layout="wide")#should be run first before any other streamlit command

    benchmark=st.selectbox("MCQ & SAQ",["Unexplained", "Explained", "Zero Shot with Base Prompt", "Zero-Shot with Instruct Prompt", "3-Shot to 5-Shot with Instruct Prompt","MedQA Zero-Shot with Base Prompt"])
    paths={
        "Unexplained":"./data/model_noexplained.csv",
        "Explained":"./data/model_explanations.csv",
        "Zero Shot with Base Prompt":"./data/model_zero_shot.csv",
        "Zero-Shot with Instruct Prompt":"./data/model_eval_zero_shot_instruct.csv",
        "3-Shot to 5-Shot with Instruct Prompt":"./data/model_eval_3shot_instruct.csv",
        "MedQA Zero-Shot with Base Prompt":"./data/model_eval_zero_shot_base2.csv",
    }

    df=pd.read_csv(paths[benchmark])

    
    col1, col2 = st.columns(2)
    with col1:
        st.title("LLM Leaderboard")

        filter=st.multiselect("Filter by columns",df.columns.tolist())
        filter=filter if filter else df.columns.tolist()
        st.dataframe(df[filter],use_container_width=True)

    with col2:
        st.title("Analysis")

        columns={
            "Unexplained":['Afrimed-QA v1', 'USMLE', 'AfriMed-QA Experts', 'AFR-MCQ','AFR-SAQ BertScore'],
            "Explained":['MCQ Accuracy', 'MCQ BertScore F1', 'MCQ Avg Rouge','SAQ BertScore F1', 'SAQ Avg Rouge', 'Consumer Queries BertScore F1','Consumer Queries Avg Rouge'],
            "Zero Shot with Base Prompt": ['MCQ Accuracy', 'MCQ BertScore F1', 'MCQ Avg Rouge','SAQ BertScore F1', 'SAQ Avg Rouge', 'Consumer Queries BertScore F1','Consumer Queries Avg Rouge'],
            "Zero-Shot with Instruct Prompt":['MCQ Accuracy', 'MCQ BertScore F1', 'MCQ Avg Rouge','SAQ BertScore F1', 'SAQ Avg Rouge', 'Consumer Queries BertScore F1','Consumer Queries Avg Rouge'],
            "3-Shot to 5-Shot with Instruct Prompt":['MCQ Accuracy', 'MCQ BertScore F1', 'MCQ Avg Rouge','SAQ BertScore F1', 'SAQ Avg Rouge', 'Consumer Queries BertScore F1','Consumer Queries Avg Rouge'],
            "MedQA Zero-Shot with Base Prompt":[ 'MCQ Accuracy', 'MCQ BertScore F1', 'MCQ Avg Rouge','SAQ BertScore F1', 'SAQ Avg Rouge', 'Consumer Queries BertScore F1','Consumer Queries Avg Rouge'],
        }

        filter=st.selectbox("Filter by benchmark",columns[benchmark])
        max_index = df[filter].idxmax()

        # Assign colors: highlight max in green, others in light gray
        colors = ["lightgray"] * len(df)
        colors[max_index] = "orange"

        # Plot
        fig = go.Figure(data=[
            go.Bar(
                x=df["Model Name"],
                y=df[filter],
                marker_color=colors
            )
        ])
        fig.update_layout(title=f"<b><span style='color:orange'>{df["Model Name"][max_index]}</span></b>  was the best model for {filter} ",)

        st.plotly_chart(fig, use_container_width=True)


    

        
        

if __name__ == "__main__":
    main()