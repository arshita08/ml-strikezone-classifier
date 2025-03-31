import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pybaseball import statcast_batter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Optional: suppress warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# 📌 Batter IDs
players = {
    "Aaron Judge (6'7\")": 592450,
    "Jose Altuve (5'6\")": 514888,
    "David Ortiz (6'3\")": 120074
}

# 🧠 Page Title
st.title("⚾ Strike Zone Classifier")
st.markdown("Predict whether a pitch is a **strike or ball** based on player-specific pitch location data.")

# 🎯 Player selection
player_name = st.selectbox("Choose a Player:", list(players.keys()))
batter_id = players[player_name]

# ⚙️ Hyperparameters
C = st.slider("C (Regularization)", 0.1, 10.0, 1.0, 0.1)
gamma = st.slider("Gamma", 0.01, 10.0, 1.0, 0.1)

# 📦 Load & clean data
@st.cache_data
def load_data(batter_id):
    df = statcast_batter('2017-01-01', '2017-12-31', batter_id)
    df = df[['plate_x', 'plate_z', 'type']]
    df = df.dropna()
    df = df[df['type'].isin(['S', 'B'])]
    df['type'] = df['type'].map({'S': 1, 'B': 0})
    return df

df = load_data(batter_id)

# 🧠 Model training
X = df[['plate_x', 'plate_z']]
y = df['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = SVC(kernel='rbf', C=C, gamma=gamma)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# 🎯 Strike zone plot
def plot_strike_zone(X, y, model, title):
    x_min, x_max = X['plate_x'].min() - 1, X['plate_x'].max() + 1
    z_min, z_max = X['plate_z'].min() - 1, X['plate_z'].max() + 1
    xx, zz = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(z_min, z_max, 500))
    mesh_points = np.c_[xx.ravel(), zz.ravel()]
    predictions = model.predict(mesh_points).reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, zz, predictions, alpha=0.25, cmap=plt.cm.coolwarm)
    plt.scatter(X['plate_x'], X['plate_z'], c=y, cmap=plt.cm.coolwarm, edgecolors='k', alpha=0.6)
    plt.title(title)
    plt.xlabel("Plate X (horizontal)")
    plt.ylabel("Plate Z (vertical)")
    plt.xlim(-2.5, 2.5)
    plt.ylim(0, 5)
    st.pyplot()

# 🔍 Display results
st.subheader(f"Model Accuracy: {accuracy:.2%}")
plot_strike_zone(X, y, model, f"{player_name} Strike Zone")
