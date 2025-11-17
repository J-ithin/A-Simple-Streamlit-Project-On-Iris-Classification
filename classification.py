import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import random

# -------------------------------
# 🎯 Page Configuration
# -------------------------------
st.set_page_config(page_title="🌺 Iris Intelligence Dashboard", layout="wide")

st.title("🌺 **Iris Intelligence Dashboard**")
st.caption("Experience AI-powered Iris flower prediction with visual insights and fun facts 🌸")

# -------------------------------
# 📦 Load Data
# -------------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# -------------------------------
# ⚙️ Model Training
# -------------------------------
X = df.iloc[:, :-1]
y = df['species']
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# -------------------------------
# 🌿 Sidebar User Inputs
# -------------------------------
st.sidebar.header("🌿 Input Flower Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()))

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# -------------------------------
# 🔮 Prediction and Confidence
# -------------------------------
prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]
confidence = np.max(probabilities) * 100
predicted_species = target_names[prediction]

# -------------------------------
# 🌸 Prediction Output
# -------------------------------
st.markdown("## 🌸 Prediction Result")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"### 🪷 Predicted Species: **{predicted_species}**")
    st.progress(int(confidence))
    st.write(f"**Model Confidence:** {confidence:.2f}%")

    if confidence < 70:
        st.warning("🤔 The model is unsure — values may be overlapping between species.")
    elif confidence < 90:
        st.info("😊 The prediction looks reliable.")
    else:
        st.success("🌟 The model is very confident!")

with col2:
    flower_colors = {"setosa": "pink", "versicolor": "purple", "virginica": "violet"}
    flower_color = flower_colors.get(predicted_species.lower(), "green")

    fig, ax = plt.subplots(figsize=(3, 3))
    circle = plt.Circle((0.5, 0.5), 0.3, color=flower_color, alpha=0.6)
    ax.add_artist(circle)
    ax.text(0.5, 0.5, predicted_species, ha="center", va="center", fontsize=14, color="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    st.pyplot(fig)

# -------------------------------
# 📊 Feature Radar Chart
# -------------------------------
st.subheader("📈 Feature Radar Visualization")

features = X.columns
values = input_data[0]

fig_radar = plt.figure(figsize=(5, 5))
categories = list(features)
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
values = np.concatenate((values, [values[0]]))
angles += angles[:1]

ax = plt.subplot(111, polar=True)
ax.fill(angles, values, color="violet", alpha=0.25)
ax.plot(angles, values, color="purple", linewidth=2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, color="gray", size=10)
st.pyplot(fig_radar)



# -------------------------------
# 💡 Random Fun Facts
# -------------------------------
fun_facts = [
    "🌺 The Iris flower is named after the Greek goddess of the rainbow.",
    "🌼 There are over 300 species of Iris found worldwide.",
    "💧 Irises can grow in deserts, swamps, and even cold regions!",
    "🎨 The Iris was a favorite subject for artist Vincent van Gogh.",
    "🌿 The three petals of an Iris symbolize faith, wisdom, and valor."
]

if st.button("✨ Show a Fun Iris Fact"):
    st.info(random.choice(fun_facts))

st.markdown("---")


