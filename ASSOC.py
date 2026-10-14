# app.py - Streamlit interface ya Online Retail Association Rules
# Endesha kwa:  streamlit run app.py

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# 1. MIPANGILIO YA UKURASA
# ---------------------------------------------------------
st.set_page_config(page_title="Online Retail Recommender", layout="wide")

# ---------------------------------------------------------
# 2. PAKIA MODEL (bila kufundisha upya)
# @st.cache_resource = pakia mara moja tu, isirudiwe kila click
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    rules = joblib.load("association_rules.joblib")
    items = joblib.load("item_list.joblib")
    return rules, items

rules, item_list = load_model()

# ---------------------------------------------------------
# 3. SIDEBAR: navigation links
# ---------------------------------------------------------
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Chagua ukurasa:",
    ["Online Retail Manual Entry", "Online Retail Upload Form"],
)

# Slider ya lift ya chini kabisa (inatumika kwenye kurasa zote mbili)
min_lift = st.sidebar.slider("Lift ya chini kabisa", 1.0, 15.0, 3.0, 0.5)
st.sidebar.caption("Lift ya juu = bidhaa zinahusiana kwa nguvu zaidi.")

# Chuja rules zenye lift ya juu tu
good_rules = rules[rules["lift"] >= min_lift]


# ---------------------------------------------------------
# 4. FUNCTION: pata mapendekezo kutoka kwa bidhaa zilizopo
# ---------------------------------------------------------
def recommend(basket_items):
    """basket_items = set ya bidhaa. Inarudisha DataFrame ya mapendekezo."""
    results = []
    for _, row in good_rules.iterrows():
        # Rule inatumika kama bidhaa zote za antecedents ziko kwenye kapu
        if set(row["antecedents"]).issubset(basket_items):
            for product in row["consequents"]:
                if product not in basket_items:      # usipendekeze alichonacho tayari
                    results.append({
                        "Recommended Product": product,
                        "Because of": ", ".join(row["antecedents"]),
                        "Confidence": round(row["confidence"], 2),
                        "Lift": round(row["lift"], 2),
                    })
    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results)
    # Kama bidhaa imependekezwa mara nyingi, acha yenye lift kubwa zaidi
    out = out.sort_values("Lift", ascending=False)
    out = out.drop_duplicates(subset="Recommended Product")
    return out.reset_index(drop=True)


# ---------------------------------------------------------
# 5. UKURASA WA KWANZA: Manual Entry
# ---------------------------------------------------------
if page == "Online Retail Manual Entry":
    st.title("Online Retail Manual Entry")
    st.write("Andika jina la bidhaa (au sehemu ya jina) upate mapendekezo.")

    text = st.text_input("Bidhaa:", placeholder="mfano: TEACUP")

    if text:
        # Tafuta bidhaa zenye neno alilandika (bila kujali herufi kubwa/ndogo)
        matches = [i for i in item_list if text.strip().upper() in i]

        if not matches:
            st.warning("Hakuna bidhaa iliyopatikana. Jaribu neno lingine.")
        else:
            item = st.selectbox("Chagua bidhaa kamili:", matches)
            recs = recommend({item})

            if recs.empty:
                st.info(f"Hakuna mapendekezo yenye lift >= {min_lift} kwa '{item}'.")
            else:
                st.success(f"Wateja wanaonunua '{item}' mara nyingi hununua pia:")
                st.dataframe(recs, width="stretch")

# ---------------------------------------------------------
# 6. UKURASA WA PILI: Upload Form
# ---------------------------------------------------------
else:
    st.title("Online Retail Upload Form")
    st.write("Pakia CSV yenye columns: **CustomerID, InvoiceDate, Description**")

    file = st.file_uploader("Chagua faili la CSV", type="csv")

    if file is not None:
        data = pd.read_csv(file, encoding="latin1")

        # Hakikisha columns zinazohitajika zipo
        needed = ["CustomerID", "InvoiceDate", "Description"]
        missing = [c for c in needed if c not in data.columns]
        if missing:
            st.error(f"Columns hazipo: {missing}")
            st.stop()

        # Safisha data kwa njia ile ile kama wakati wa training
        data = data[needed].dropna()
        data["Description"] = data["Description"].str.strip().str.upper()
        data["CustomerID"] = data["CustomerID"].astype(int)
        data["Date"] = pd.to_datetime(data["InvoiceDate"]).dt.date

        # Kapu moja = mteja + siku
        sessions = data.groupby(["CustomerID", "Date"])["Description"].apply(set)

        # Tafuta mapendekezo kwa kila session
        rows = []
        for (cust, date), basket_items in sessions.items():
            recs = recommend(basket_items)
            for _, r in recs.iterrows():
                rows.append({"CustomerID": cust, "Date": date, **r.to_dict()})

        st.write(f"Sessions zilizopatikana: **{len(sessions)}**")

        if not rows:
            st.info("Hakuna mapendekezo yaliyopatikana kwa data hii.")
        else:
            result = pd.DataFrame(rows)
            st.success(f"Mapendekezo {len(result)} yamepatikana.")
            st.dataframe(result, width="stretch")
            st.download_button("Pakua matokeo (CSV)",
                               result.to_csv(index=False),
                               "recommendations.csv")