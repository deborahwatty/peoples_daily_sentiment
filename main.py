import streamlit as st
import pandas as pd
import os
from sentiment_assistant import SentimentAssistant


# Helper function to load data
@st.cache_data
def load_data(year):
    return pd.read_parquet(f'parquet-new/{year}.parquet')


# Title of the app
st.title('Keyword Search and Sentiment Analysis')

# Dropdown menus for choosing time frame
years = list(range(1946, 2017))
from_year = st.selectbox('From Year', years, index=0)
to_year = st.selectbox('To Year', years, index=len(years) - 1)

if from_year > to_year:
    st.error('"From Year" must be before "To Year"')
else:
    # Input for keyword
    keyword = st.text_input('Enter keyword')

    if keyword:
        # Load data for the selected years
        data_dict = {}
        for year in range(from_year, to_year + 1):
            data_dict[year] = load_data(year)

        # Filter dataframes based on keyword
        filtered_dfs = []
        for year, df in data_dict.items():
            filtered_df = df[
                df['title'].str.contains(keyword, case=False, na=False) | df['doc_content'].str.contains(keyword,
                                                                                                         case=False,
                                                                                                         na=False)]
            if not filtered_df.empty:
                filtered_dfs.append(filtered_df)

        if filtered_dfs:
            result_df = pd.concat(filtered_dfs)
            st.write(f'Found {len(result_df)} entries containing "{keyword}"')

            # Input for sample size
            sample_size = st.number_input('Enter sample size (up to 100)', min_value=1, max_value=100, value=20)

            # Button to sample articles
            if st.button("Sample Articles"):
                if len(result_df) > sample_size:
                    sampled_df = result_df.sample(sample_size)
                else:
                    sampled_df = result_df.copy()

                # Store the sample dataframe in the session state
                st.session_state.sampled_df = sampled_df
                st.session_state.current_choice = None  # Reset current choice to allow for new analysis

            # User choice between "openai" and "gemini"
            if 'sampled_df' in st.session_state:
                # Display the resulting DataFrame
                st.write(f'Sampled {len(st.session_state.sampled_df)} entries for analysis')
                st.dataframe(st.session_state.sampled_df)

                choice = st.radio('Choose Sentiment Analysis Model', ['openai', 'gemini'], index=0)

                # Check if the model has already been run
                if 'results' not in st.session_state or st.session_state.current_choice != choice:

                    # Button to run sentiment analysis
                    if st.button('Run Sentiment Analysis'):
                        st.session_state.current_choice = choice
                        with st.spinner('Running sentiment analysis...'):
                            # Initialize SentimentAssistant
                            sa = SentimentAssistant(llm=choice, num_entities=1,
                                                    df=st.session_state.sampled_df)  # Changed from filepath to df

                            # Run sentiment analysis
                            if choice == 'openai':
                                result = sa.run_single_openai_fulldf(keyword)
                            else:
                                result = sa.run_single_gemini_fulldf(keyword)

                            # Store results in session state
                            st.session_state.results = result


                            score_column = f"{keyword}_{choice}_score"

                            # Create a mapping of filenames to labels
                            filename_to_label = {filename: data['label'] for filename, data in result.items()}

                            # Assign labels to the score column in the sampled_df
                            st.session_state.sampled_df[score_column] = st.session_state.sampled_df['filename'].map(
                                filename_to_label)

                            # Display the updated DataFrame with sentiment scores
                            st.write(f'DataFrame with Score')
                            st.dataframe(st.session_state.sampled_df)

                            # Plot the result
                            fig = sa.plot(result, keyword)
                            st.pyplot(fig)
                else:
                    # Show existing results and plot
                    if 'results' in st.session_state:
                        result = st.session_state.results
                        fig = SentimentAssistant(llm=choice, num_entities=1, df=st.session_state.sampled_df).plot(
                            result, keyword)
                        st.pyplot(fig)

                        st.write(f'DataFrame with Score')
                        st.dataframe(st.session_state.sampled_df)
                    else:
                        st.warning('Please run sentiment analysis first.')
            else:
                st.warning('Please sample articles first.')
        else:
            st.write(f'No entries found containing "{keyword}"')

