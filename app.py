import streamlit as st
import numpy as np
from pandas import DataFrame
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer


# For Flair (Keybert)
from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns

# For download buttons
from functionforDownloadButtons import download_button
import os
import json

list_keywords = ['climate','environment','waste','renewable','water', 'carbon',
           'sustainab','strategy','scope','emission','ghg', 'co2',
           'greenhouse','target', 'environment', 
            
            'board overs','board overs', 'committ',
            
            
            'climate','environment','waste','natural resource','water'
            
            'flood','warm','wind','flood','fuel','disaster', 'warm', 'hurricane', 'heat','solar', 'coal',
            'catastroph', 
            
            'power','electric','green','energy','air',
            'catastroph','physical','transition','solar','coal','net-zero','net zero', 'catastroph',
            'hurricane','terrorism','storm','tornado','severe','weather','IPCC','°','warm','land','ocean'
            
            'ghg', 'emission', 'greenhouse', 'co2', 'scope 1', 'scope 2', 'scope 3', 'carbon',
            
            'reduc','reduction', 'science', 'Paris Agreement', 'net-zero','net zero', 'committ'
           ]

# set page title and icon
st.set_page_config(
    page_title="BERT Keyword Extractor",
    # page_icon="🎈",
)

# set app layout width
def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()





with st.expander("About this app", expanded=True):

    st.write(
        """
	Includes example article from https://www.db.com/what-we-do/responsibility/. Please change the text for more example outputs.
	    """
    )

    st.markdown("")



st.markdown("## ** Paste document **")


with st.form(key="my_form"):

    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        # Model type
        ModelType = st.radio(
            "Choose your model",
            ["Custom Model", "Default"],
            # help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",
        )

        if ModelType == "Default (Our BERT Model)":

            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT("sentence-transformers/all-MiniLM-L6-v2")
                # return KeyBERT(model=roberta)

            kw_model = load_model()

        else:

            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT("sentence-transformers/all-MiniLM-L6-v2")
                # return KeyBERT("distilbert-base-nli-mean-tokens")

            kw_model = load_model()

        top_N = st.slider(
            "# of results",
            min_value=1,
            max_value=30,
            value=30,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 30.",
        )
        min_Ngrams = st.number_input(
            "Minimum Ngram",
            min_value=1,
            max_value=4,
            help="""The minimum value for the ngram range.

*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.

To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
        )

        max_Ngrams = st.number_input(
            "Maximum Ngram",
            value=2,
            min_value=1,
            max_value=4,
            help="""The maximum value for the keyphrase_ngram_range.

*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.

To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
        )

        StopWordsCheckbox = st.checkbox(
            "Remove stop words",  value=True,
            help="Tick this box to remove stop words from the document (currently English only)",
        )

        use_MMR = st.checkbox(
            "Use MMR",
            value=True,
            help="You can use Maximal Margin Relevance (MMR) to diversify the results. It creates keywords/keyphrases based on cosine similarity. Try high/low 'Diversity' settings below for interesting variations.",
        )

        Diversity = st.slider(
            "Keyword diversity (MMR only)",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help="""The higher the setting, the more diverse the keywords.
            
Note that the *Keyword diversity* slider only works if the *MMR* checkbox is ticked.

""",
        )
    
#     data = """We formalized a sustainability function to continue holding us accountable to the corporate governance standards that pillar responsible business. In summary, to ensure climate resilience, our Board, executives, and senior leadership monitor changing customer tastes and demands, regulatory requirements, as well as other impacts to our business. Our board oversight for ESG is a part of the Nominating and Corporate Governance Committees responsibility, and oversight specific to human capital and diversity, equity, and inclusion matters is delegated to the Human Capital and Compensation Committee. The ESG Committee meets regularly to address issues such as energy efficiency, carbon emissions, and environmental risks and opportunities in our homes and operations. The process to identify, manage, and integrate climate risk is embedded in our Enterprise Risk Assessment program, which is overseen by our Chief Financial Officer (CFO). The Human Capital and Compensation Committee of the board of trustees oversees our companys human capital programs and policies, including with respect to employee retention and development, and regularly meets with senior management to discuss these issues. This information is brought to our ESG Committee and discussed to help further develop our ESG strategy. Board oversight of ESG:, we formalized Board oversight for ESG as part of committee responsibilities, which we put into practice throughout. The Board is responsible for overseeing the companys approach to major risks and policies for assessing and managing these risks. The Nominating and Corporate Governance Committee has overall responsibility for our ESG program with specific topics overseen by the other Board committees. This council, led by members of senior management, provides participants with a unique forum to provide feedback from all levels in the organization. Quarterly cybersecurity reviews are led by our Chief Technology Officer and Vice President of Information Security, who leads our dedicated cybersecurity team, and other members of our executive leadership team, including our CEO, CFO, and CLO. In summary, to ensure climate resilience, our Board, executives, and senior leadership monitor changing customer tastes and demands, regulatory requirements, as well as other impacts to our business. we formalized a sustainability function to continue holding us accountable to the corporate governance standards that pillar responsible business. The Audit Committee and our Board also conduct a full review of cybersecurity annually. The Senior Vice President of Sustainability, AMH Development, and Property Operations teams collaborate to oversee the strategic implementation our environmental and energy programs. Board oversight of ESG:, we formalized Board oversight for ESG as part of committee responsibilities, which we put into practice throughout 2021. The Human Capital and Compensation Committee oversees our programs on talent, leadership and culture, which include diversity, equity and inclusion. Management role. ESG Committee is responsible for supporting the companys efforts in developing, implementing, monitoring, and reporting on environmental initiatives including those relevant to climate change. 
#      """,

#     data="""
# 	Deutsche Bank has signed up to the World Green Building Council’s (WGBC) Net Zero Carbon Buildings Commitment, pledging to reduce and compensate operational emissions associated with energy used to light, heat, cool and power buildings, for assets over which it has direct control.

# 	In addition, as part of the commitment, the bank will maximise reduction of the embodied carbon emissions for owned new developments and major renovations by 2030 and compensate all residual upfront emissions. Embodied carbon emissions are those associated with the manufacture of materials, transport to and from the construction site, and processes used during the construction phase of a building or infrastructure.

# 	Together, operational and embodied carbon emissions combined are 39 percent of global energy-related carbon emissions (28 percent from operational emissions, 11 percent from construction, according to the WGBC).

# 	Deutsche Bank is one of seven new signatories to the Commitment, bringing the total to 169 signatories over 28 cities in six states and regions.

# 	The WGBC is a global action network comprised of around 70 Green Building Councils around the globe, and as members of the UN Global Compact, they work with businesses, organisations and governments to support the ambitions of the Paris Agreement and the UN’s Sustainable Development Goals (SDGs). Their aim is to transform the building and construction sector across three strategic areas: climate action; health and wellbeing; and resources and circularity. 

# 	The Net Zero Carbon Buildings Commitment was launched in September 2018 at the Global Climate Action Summit and is part of WGBC’s global “Advancing Net Zero” project to accelerate the path to net zero carbon buildings to 100 percent by 2050.
# 	Deutsche Bank now also member of EP100

# 	Deutsche Bank has also become a member of the EP100 initiative. This is a global corporate Energy Productivity initiative led by the international non-profit Climate Group.

# 	EP100 members are committed to doubling their energy productivity, rolling out energy management systems, or achieving net zero carbon buildings, all within set timeframes. Over 120 businesses have already committed to measuring and reporting on energy efficiency improvements, essential for the reduction in energy-related emissions needed to achieve global climate goals. 
# 	""",

data="""
The Company also considers climate change risks in making investment decisions to acquire and develop properties.  As part of our proactive measures to increase awareness and preparedness for the potential future impact from climate change, the Company completed resilience assessments of our Boston portfolio and our Los Angeles portfolio.  Through these assessments, we have developed a strategy and methodology for assessing climate resilience that will be expanded in 2022 to our full portfolio and potential investment opportunities.  In addition, the Company partners with cities and local governments, through organizations like the Boston Green Ribbon Commission, to develop climate resilience plans and strategies for fighting climate change.  We aim to make climate resilience a key driver in our overall business strategy in order to mitigate risks and identify long-term value creation opportunities. 
Through this screening process, we weigh several sustainability characteristics that contribute to long-term value and resilience.  We conduct energy audits and identify potential opportunities to increase efficiency in building systems.  This includes considering LEED status, on-site clean and renewable energy, energy intensity, benchmarking our energy use and scanning for lighting retrofits and central system controls.  We also consider physical risks such as the potential for flooding, wildfires and environmental hazards and conduct a Phase I Environmental Site Assessment on all new acquisitions. 
2 Equity Residential used the Intergovernmental Panel on Climate Change (IPCC) Representative Concentration Pathway (RCP) 4. 5 and a “businesses-as usual” scenario of RCP 8. 5 as the two scenarios to assess impacts and selected medium- and longer-term timeframes of 2030 and 2050 based on the types of expected hazards and regulatory frameworks impacting the Boston area. 
Environment—Climate Strategy and Portfolio Resilience;Sustainable Buildings;Energy and Emissions Describe the resilience of the organizations strategy, taking into consideration different climate-related scenarios, including a 2°C or lower scenario.
	""",
    with c2:
        data = st.text_area(
            "Paste your text below (max 1000 words)",value="\n".join(data),
            height=510,
        )

        MAX_WORDS = 1000

        import re

        res = len(re.findall(r"\w+", data))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 500 words will be reviewed"
            )

            data = data[:MAX_WORDS]

        submit_button = st.form_submit_button(label="Get me the data!")

    if use_MMR:
        mmr = True
    else:
        mmr = False

    if StopWordsCheckbox:
        StopWords = "english"
    else:
        StopWords = None

if not submit_button:
    st.stop()

if min_Ngrams > max_Ngrams:
    st.warning("min_Ngrams can't be greater than max_Ngrams")
    st.stop()

keywords = kw_model.extract_keywords(
    data,
    keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
    use_mmr=mmr,
    stop_words=StopWords,
    top_n=top_N,
    diversity=Diversity,
    vectorizer=KeyphraseCountVectorizer()
)

st.markdown("## ** Check & download results **")

st.header("")

cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

with c1:
    CSVButton2 = download_button(keywords, "Data.csv", "🎁 Download (.csv)")
with c2:
    CSVButton2 = download_button(keywords, "Data.txt", "🎁 Download (.txt)")
with c3:
    CSVButton2 = download_button(keywords, "Data.json", "🎁 Download (.json)")

st.header("")

df = (
    DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
    .sort_values(by="Relevancy", ascending=False)
    .reset_index(drop=True)
)

df=df[df['Keyword/Keyphrase'].str.contains('|'.join(list_keywords))]

df.index += 1


# Add Styling to the table columns and rows

cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Relevancy",
    ],
)

c1, c2, c3 = st.columns([1, 3, 1])

format_dictionary = {
    "Relevancy": "{:.1%}",
}

df = df.format(format_dictionary)

with c2:
    st.table(df)
