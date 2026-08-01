# Import packages
import pickle
from dash import Dash, html, dcc, Input, Output, ClientsideFunction
import dash_ag_grid as dag
import pandas as pd
import plotly.express as px
from dash_holoniq_wordcloud import DashWordcloud
# Incorporate data

with open('emergence_report.pkl', 'rb') as f:
    emergence_report = pickle.load(f)

df = emergence_report[emergence_report['acceleration'] > 100].head(15).copy()

df["emergence_score"] = df["emergence_score"]*10000

# Initialize the app
app = Dash()

# ---- Methodology / "how this works" copy -----------------------------------
# Matches the 4-stage pipeline: fetch_arxiv -> extract_phrases -> cluster_phrases
# -> trend_analysis. Edit the "Scoring emergence" bullet if your emergence_report
# defines emergence_score / acceleration differently than described here.
PIPELINE_EXPLANATION = """
**1. Collecting the papers**
Metadata (title, abstract, categories, submission date) is pulled from the
arXiv API for high-energy physics categories — `hep-ph`, `hep-th`, `hep-ex`,
`hep-lat` — over a multi-year window. Large date ranges are split into
smaller slices to stay within arXiv's API limits, with automatic retries and
resume support if a long-running fetch gets interrupted.

**2. Extracting topic phrases**
Each paper's title and abstract are parsed with spaCy to pull out noun
phrases (2-5 words) — e.g. *"axion-like particle"*, *"direct detection of
dark matter"*. Generic phrases (*"this paper"*, *"our approach"*) are
filtered out, since they don't describe a research topic.

**3. Grouping near-duplicate phrasings**
The same idea gets written many different ways across papers. This stage
groups phrasings that mean the same thing into a single canonical topic —
e.g. *"dark matter direct detection"* and *"direct detection experiments for
dark matter"* collapse into one. Phrases are grouped either by text
similarity or by semantic embedding similarity, depending on how the
clustering stage is configured.

**4. Scoring emergence**
For each topic, we track its *share* of that year's papers (not raw mention
count — since arXiv submission volume itself grows every year, raw counts
go up for almost everything regardless of real relevance). `emergence_score`
compares a topic's recent share against its earlier baseline share, so a
higher score means the topic occupies a growing fraction of the literature.
`acceleration` goes a step further and captures whether that growth is
*itself* speeding up, surfacing topics that are still in an early, steepening
takeoff rather than ones that grew once and have since leveled off.

**This view**
The word cloud below shows the top 15 topics filtered to the highest
acceleration, sized by `emergence_score`. Click any word to open an
Inspire-HEP search for well-cited (10+ citations) papers on that topic.
"""

# App layout
app.layout = html.Div([
    html.Div([
        html.Details([
            html.Summary(
                "How this pipeline works",
                style={
                    'cursor': 'pointer',
                    'color': '#dcdc1d',
                    'fontSize': '18px',
                    'fontWeight': 'bold',
                    'padding': '10px',
                }
            ),
            dcc.Markdown(
                PIPELINE_EXPLANATION,
                style={'color': '#e8e8e8', 'lineHeight': '1.5'}
            ),
        ], style={
            'backgroundColor': '#001f00',
            'border': '1px solid #dcdc1d',
            'borderRadius': '6px',
            'padding': '10px 20px',
            'marginBottom': '20px',
            'maxWidth': '1100px',
        }),
    ]),
    html.Div([
        DashWordcloud(
            id='wordcloud-input',
            list=df[['field_name', 'emergence_score']].values.tolist(),
            width=1100, height=600,
            gridSize=20,
            color="#dcdc1d",
            backgroundColor='#001f00',
            shuffle=False,
            rotateRatio=0.5,
            shrinkToFit=True,
            shape='circle',
            hover=True
            )
        ]),
    html.Div(id='dummy-output', style={'display': 'none'})
    ])

app.clientside_callback(
    """
    function(click_data) {
        if (click_data) {
            // click_data returns [word, size]
            const word = click_data[0];
            const url = "https://inspirehep.net/literature?sort=mostrecent&q=topcite%2010%2B%20" + encodeURIComponent(word);
            window.open(url, '_blank');
        }
        return '';
    }
    """,
    Output('dummy-output', 'children'),
    Input('wordcloud-input', 'click')
)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
