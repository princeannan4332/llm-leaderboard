import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def main():
    st.set_page_config(layout="wide")  # should be run first

    benchmark = st.selectbox(
        "MCQ & SAQ",
        [
            "Unexplained",
            "Explained",
            "Zero Shot with Base Prompt",
            "Zero-Shot with Instruct Prompt",
            "3-Shot to 5-Shot with Instruct Prompt",
            "MedQA Zero-Shot with Base Prompt",
        ],
    )

    paths = {
        "Unexplained": "./data/model_noexplained.csv",
        "Explained": "./data/model_explanations.csv",
        "Zero Shot with Base Prompt": "./data/model_zero_shot.csv",
        "Zero-Shot with Instruct Prompt": "./data/model_eval_zero_shot_instruct.csv",
        "3-Shot to 5-Shot with Instruct Prompt": "./data/model_eval_3shot_instruct.csv",
        "MedQA Zero-Shot with Base Prompt": "./data/model_eval_zero_shot_base2.csv",
    }

    columns = {
        "Unexplained": ['Afrimed-QA v1', 'USMLE', 'AfriMed-QA Experts', 'AFR-MCQ','AFR-SAQ BertScore'],
        "Explained": ['MCQ Accuracy', 'MCQ BertScore F1', 'MCQ Avg Rouge','SAQ BertScore F1', 'SAQ Avg Rouge', 'Consumer Queries BertScore F1','Consumer Queries Avg Rouge'],
        "Zero Shot with Base Prompt": ['MCQ Accuracy', 'MCQ BertScore F1', 'MCQ Avg Rouge','SAQ BertScore F1', 'SAQ Avg Rouge', 'Consumer Queries BertScore F1','Consumer Queries Avg Rouge'],
        "Zero-Shot with Instruct Prompt": ['MCQ Accuracy', 'MCQ BertScore F1', 'MCQ Avg Rouge','SAQ BertScore F1', 'SAQ Avg Rouge', 'Consumer Queries BertScore F1','Consumer Queries Avg Rouge'],
        "3-Shot to 5-Shot with Instruct Prompt": ['MCQ Accuracy', 'MCQ BertScore F1', 'MCQ Avg Rouge','SAQ BertScore F1', 'SAQ Avg Rouge', 'Consumer Queries BertScore F1','Consumer Queries Avg Rouge'],
        "MedQA Zero-Shot with Base Prompt": ['MCQ Accuracy', 'MCQ BertScore F1', 'MCQ Avg Rouge','SAQ BertScore F1', 'SAQ Avg Rouge', 'Consumer Queries BertScore F1','Consumer Queries Avg Rouge'],
    }

    df = pd.read_csv(paths[benchmark])

    col1, col2 = st.columns(2)

    with col1:
        st.title("LLM Leaderboard")
        filter = st.multiselect("Filter by columns", df.columns.tolist())
        filter = filter if filter else df.columns.tolist()
        st.dataframe(df[filter], use_container_width=True)

    with col2:
        st.title("Analysis")
        mode = st.selectbox("Select Mode", ["Compare All LLMs Across  Specific Datasets Benchmark", "Compare Specific LLMs Across Specific Datasets Benchmark"])

        if mode == "Compare All LLMs Across  Specific Datasets Benchmark":
            filter = st.selectbox("Filter by benchmark", columns[benchmark])
            max_index = df[filter].idxmax()

            # Assign colors
            colors = ["lightgray"] * len(df)
            colors[max_index] = "orange"

            fig = go.Figure(data=[
                go.Bar(
                    x=df["Model Name"],
                    y=df[filter],
                    marker_color=colors
                )
            ])
            fig.update_layout(
                title=f"<b><span style='color:orange'>{df['Model Name'][max_index]}</span></b> was the best model for {filter}",
                xaxis_title="Model Name",
                yaxis_title=filter,
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            headers = df["Model Name"].tolist()
            models = st.multiselect("Select the models", headers)
            selected_columns = st.multiselect("Filter by benchmarks", columns[benchmark])

            if models and selected_columns:
                filtered_df = df[df["Model Name"].isin(models)]

                fig = go.Figure()
                for col in selected_columns:
                    fig.add_bar(
                        name=col,
                        x=filtered_df["Model Name"],
                        y=filtered_df[col],
                        text=filtered_df[col],
                        textposition='auto'
                    )

                fig.update_layout(
                    barmode='group',
                    title='Model Comparison Across Selected Metrics',
                    xaxis_title='Model Name',
                    yaxis_title='Score',
                    yaxis=dict(tickformat=".2f")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select both models and metrics to display the grouped chart.")

if __name__ == "__main__":
    main()
