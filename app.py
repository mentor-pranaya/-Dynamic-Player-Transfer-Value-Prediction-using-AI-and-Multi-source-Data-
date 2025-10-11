import streamlit as st
import requests
import os
import pandas as pd

# Load dataset for player selection and defaults
try:
    DF = pd.read_csv('final_data.csv')
except Exception:
    DF = None

st.set_page_config(page_title='TransferIQ - Predict', layout='wide')

st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to:', ['Home','New Player Prediction'])

if page == 'Home':
    st.title('TransferIQ: Dynamic Player Transfer Value Prediction')
    st.write('Small prototype that predicts player market value using a saved model.')
    st.write('Data source: sample `final_data.csv` in workspace.')

if page == 'New Player Prediction':
    st.header('Predict Transfer Value for a New Player')

    # Show top players in sidebar for quick reference (if dataset available)
    if DF is not None:
        top = DF[['name','current_value']].dropna()
        if 'current_value' in top.columns:
            top = top.sort_values('current_value', ascending=False).head(20)
            st.sidebar.markdown('### Top players by current value')
            st.sidebar.dataframe(top.reset_index(drop=True))

    # Player selection to prefill inputs
    player_list = ['-- New Player --']
    if DF is not None and 'name' in DF.columns:
        player_list += DF['name'].fillna('').astype(str).tolist()

    selected_player = st.selectbox('Select existing player to prefill fields', player_list)

    # defaults
    default = {
        'age': 25,
        'appearance': 10,
        'goals': 1,
        'assists': 0,
        'minutes played': 900,
        'days_injured': 0,
        'games_injured': 0,
        'current_value': 1000000,
        'highest_value': 2000000,
        'position_encoded': 3,
        'winger': 0
    }

    if selected_player != '-- New Player --' and DF is not None:
        row = DF[DF['name'] == selected_player]
        if not row.empty:
            r = row.iloc[0]
            try:
                default['age'] = int(r.get('age', default['age']))
            except Exception:
                pass
            for k in ['appearance','goals','assists','minutes played','days_injured','games_injured','current_value','highest_value','position_encoded','winger']:
                if k in r.index:
                    try:
                        default[k] = r[k] if pd.notna(r[k]) else default[k]
                    except Exception:
                        pass

    with st.form('player'):
        # Ensure consistent numeric types: use ints for count-like fields and floats for rates where appropriate.
        age = st.number_input('Age', min_value=15, max_value=50, value=int(default.get('age', 25)), step=1)
        appearance = st.number_input('Appearances', min_value=0, value=int(default.get('appearance', 0)), step=1)
        # goals and assists can be fractional in some datasets; keep them as floats but provide float defaults
        goals = st.number_input('Total Goals', min_value=0.0, value=float(default.get('goals', 0.0)), step=0.5, format="%.1f")
        assists = st.number_input('Total Assists', min_value=0.0, value=float(default.get('assists', 0.0)), step=0.5, format="%.1f")
        minutes_played = st.number_input('Minutes Played', min_value=0, value=int(default.get('minutes played', 0)), step=1)
        days_injured = st.number_input('Days Injured', min_value=0, value=int(default.get('days_injured', 0)), step=1)
        games_injured = st.number_input('Games Injured', min_value=0, value=int(default.get('games_injured', 0)), step=1)
        current_value = st.number_input('Previous Market Value (€)', min_value=0, value=int(default.get('current_value', 0)), step=1000)
        highest_value = st.number_input('Highest Value (€)', min_value=0, value=int(default.get('highest_value', 0)), step=1000)
        pos_options = [1,2,3,4]
        try:
            pos_index = pos_options.index(int(default.get('position_encoded', 3)))
        except Exception:
            pos_index = 2
        position_encoded = st.selectbox('Position Encoded', pos_options, index=pos_index)
        winger = st.selectbox('Winger (1 yes / 0 no)', [0,1], index=int(default.get('winger', 0)))
        submitted = st.form_submit_button('Predict')

    if submitted:
        payload = {
            'features':{
                'age': age,
                'appearance': appearance,
                'goals': goals,
                'assists': assists,
                'minutes played': minutes_played,
                'days_injured': days_injured,
                'games_injured': games_injured,
                'current_value': current_value,
                'highest_value': highest_value,
                'position_encoded': position_encoded,
                'winger': winger
            }
        }
        try:
            backend_url = os.environ.get('BACKEND_URL', 'http://127.0.0.1:8000')
            resp = requests.post(f"{backend_url}/predict", json=payload, timeout=10).json()
            if 'predicted_value' in resp:
                st.success(f'Predicted Market Value: €{resp["predicted_value"]:,.0f}')
                st.markdown('---')
                st.write('Previous Value (€):', f'€{current_value:,.0f}')
                if selected_player != '-- New Player --' and DF is not None:
                    orig = DF[DF['name'] == selected_player]
                    if not orig.empty and 'current_value' in orig.columns:
                        st.write('Dataset Current Value:', f'€{float(orig.iloc[0]["current_value"]):,.0f}')

                # Attempt to fetch time-series history + sentiment for the selected player or synthesized from values
                try:
                    backend_url = os.environ.get('BACKEND_URL', 'http://127.0.0.1:8000')
                    ts_payload = {
                        'name': selected_player if selected_player != '-- New Player --' else 'new_player',
                        'current_value': current_value,
                        'highest_value': highest_value
                    }
                    ts_resp = requests.post(f"{backend_url}/predict_ts", json=ts_payload, timeout=10).json()
                    # display history and sentiment if present
                    if 'history' in ts_resp:
                        hist = ts_resp['history']
                        sent = ts_resp.get('sentiment', None)
                        import matplotlib.pyplot as plt
                        st.markdown('### Historical Market Value')
                        df_hist = pd.DataFrame({'value': hist})
                        st.line_chart(df_hist['value'])
                        if sent is not None:
                            st.markdown('### Sentiment Analysis')
                            try:
                                df_sent = pd.DataFrame({'sentiment': sent})
                                st.line_chart(df_sent['sentiment'])
                            except Exception:
                                # fallback: show raw series
                                st.write(sent)
                        if 'forecast' in ts_resp:
                            # forecast may be a list or scalar
                            fval = ts_resp['forecast']
                            try:
                                if isinstance(fval, list) and len(fval) > 0:
                                    fval = float(fval[0])
                                else:
                                    fval = float(fval)
                                st.markdown('### Forecast')
                                st.write(f"Forecasted next value: €{fval:,.0f}")
                            except Exception:
                                # show raw forecast
                                st.write(fval)

                    # Exploratory analysis: univariate & multivariate graphs using dataset if available
                    if DF is not None:
                        st.markdown('---')
                        st.markdown('## Exploratory Analysis')
                        with st.expander('Univariate analysis (distribution)'):
                            try:
                                numeric_cols = DF.select_dtypes(include=['number']).columns.tolist()
                                if not numeric_cols:
                                    st.write('No numeric columns available in dataset')
                                else:
                                    col = st.selectbox('Choose numeric feature', numeric_cols, index=numeric_cols.index('current_value') if 'current_value' in numeric_cols else 0)
                                    bins = st.slider('Bins', 5, 100, 25)
                                    hist_df = DF[col].dropna()
                                    st.write('Summary')
                                    st.write(hist_df.describe())
                                    st.markdown('Histogram')
                                    try:
                                        import matplotlib.pyplot as plt
                                        fig, ax = plt.subplots()
                                        ax.hist(hist_df, bins=bins, color='#2b8cbe', edgecolor='black')
                                        ax.set_xlabel(col)
                                        ax.set_ylabel('Count')
                                        st.pyplot(fig)
                                    except Exception:
                                        # fallback: use Streamlit's bar_chart by binning manually
                                        try:
                                            counts = pd.cut(hist_df, bins=bins).value_counts().sort_index()
                                            st.bar_chart(counts)
                                        except Exception:
                                            st.write('Could not render histogram for this feature.')
                            except Exception as e:
                                st.error('Error rendering univariate analysis: ' + str(e))

                        with st.expander('Multivariate analysis (scatter & correlation)'):
                            try:
                                opts = numeric_cols
                                if len(opts) < 2:
                                    st.write('Need at least two numeric columns for multivariate plots')
                                else:
                                    col_x = st.selectbox('X axis', opts, index=opts.index('age') if 'age' in opts else 0)
                                    col_y = st.selectbox('Y axis', opts, index=opts.index('current_value') if 'current_value' in opts else 1)
                                    color_by = st.selectbox('Color by (optional categorical)', ['None'] + [c for c in DF.columns if DF[c].dtype == 'object'], index=0)
                                    st.markdown('Scatter plot')
                                    import numpy as np
                                    plot_df = DF[[col_x, col_y]].copy()
                                    plot_df = plot_df.dropna()
                                    try:
                                        import altair as alt
                                        if color_by != 'None' and color_by in DF.columns:
                                            tmp = DF[[col_x, col_y, color_by]].copy()
                                            tmp[color_by] = tmp[color_by].fillna('')
                                            tmp = tmp.dropna()
                                            chart = alt.Chart(tmp).mark_circle(size=60).encode(x=col_x, y=col_y, color=color_by, tooltip=[col_x, col_y, color_by]).interactive()
                                        else:
                                            chart = alt.Chart(plot_df).mark_circle(size=60).encode(x=col_x, y=col_y, tooltip=[col_x, col_y]).interactive()
                                        st.altair_chart(chart, use_container_width=True)
                                    except Exception:
                                        st.write('Altair not available; falling back to simple scatter')
                                        try:
                                            import matplotlib.pyplot as plt
                                            fig, ax = plt.subplots()
                                            ax.scatter(plot_df[col_x], plot_df[col_y], alpha=0.6)
                                            ax.set_xlabel(col_x)
                                            ax.set_ylabel(col_y)
                                            st.pyplot(fig)
                                        except Exception:
                                            st.write('Could not render scatter plot')

                                    st.markdown('Correlation heatmap')
                                    try:
                                        import seaborn as sns
                                        import matplotlib.pyplot as plt
                                        import numpy as np

                                        # Allow user to limit number of features displayed to avoid clutter
                                        max_feats = st.slider('Max features to display in heatmap', min_value=6, max_value=min(40, len(opts)), value=min(20, len(opts)))
                                        display_feats = opts[:max_feats]

                                        corr = DF[display_feats].corr()
                                        # mask upper triangle
                                        mask = np.triu(np.ones_like(corr, dtype=bool))
                                        # set figure size proportional to number of features
                                        size = max(6, min(0.5 * len(display_feats), 20))
                                        fig, ax = plt.subplots(figsize=(size, size))
                                        sns.heatmap(
                                            corr,
                                            mask=mask,
                                            annot=True,
                                            fmt='.2f',
                                            cmap='coolwarm',
                                            vmax=1.0,
                                            vmin=-1.0,
                                            center=0,
                                            annot_kws={'size': 7},
                                            linewidths=0.25,
                                            cbar_kws={'shrink': 0.6},
                                            ax=ax
                                        )
                                        plt.xticks(rotation=45, ha='right')
                                        plt.yticks(rotation=0)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                    except Exception:
                                        st.write('Seaborn not available; showing correlation matrix as table')
                                        st.dataframe(DF[opts].corr())
                            except Exception as e:
                                st.error('Error rendering multivariate analysis: ' + str(e))
                except Exception:
                    # non-critical: silently skip TS if backend not available
                    st.info('Timeseries/sentiment endpoint unavailable or returned invalid data.')
            else:
                st.error('Error: ' + str(resp))
        except Exception as e:
            st.error('Could not reach backend. Start backend with uvicorn backend:app --reload')
