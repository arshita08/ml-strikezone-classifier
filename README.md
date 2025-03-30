# ⚾️ Strike Zone Classifier – MLB Pitch Classification with Machine Learning

This project builds a machine learning model that predicts whether a baseball pitch is a **strike or ball** based on its **location** over home plate. We use real pitch-by-pitch data for **Aaron Judge** to learn a personalized strike zone using **Support Vector Machines (SVM)**.

---

## 🔍 Problem Statement

Umpire strike zones vary from person to person. This project leverages MLB pitch data to:
- Model a **consistent** decision boundary
- Visualize a **personalized strike zone**
- Apply a complete **end-to-end ML pipeline**

---

## 🧠 Machine Learning Workflow

✅ **Step 1:** Load MLB pitch data from [pybaseball](https://github.com/jldbc/pybaseball)  
✅ **Step 2:** Clean and filter columns (`plate_x`, `plate_z`, `type`)  
✅ **Step 3:** Encode labels (`S` → 1, `B` → 0)  
✅ **Step 4:** Train Support Vector Machine (SVC with RBF kernel)  
✅ **Step 5:** Perform hyperparameter tuning with `GridSearchCV`  
✅ **Step 6:** Build a `Pipeline` with feature scaling  
✅ **Step 7:** Visualize the learned strike zone using decision boundaries

---

## 🛠️ Tools & Libraries

- Python + Jupyter Notebook (via Google Colab)
- [pybaseball](https://github.com/jldbc/pybaseball) for MLB pitch data
- `scikit-learn` for SVM, GridSearch, Pipelines
- `matplotlib` for visualizations
- `pandas`, `numpy` for data wrangling

---

## 📈 Results

- Achieved **~85% accuracy** on validation set
- Visualized a **player-specific strike zone**
- Used **Grid Search** to optimize SVM hyperparameters (`C`, `gamma`)
- Final model deployed through a `Pipeline`

---

## 📷 Screenshots

### 🔍 Visualization: Strike Zone for Aaron Judge  
![strikezone-demo](path-to-your-image.png)  
> *Color-coded: red = strike, blue = ball; decision boundary is SVM-predicted strike zone.*

*Tip: To add this image, take a screenshot of your Colab plot, upload it to your GitHub repo, and update `path-to-your-image.png` above.*

---

## 🚀 Future Improvements

- Compare strike zones for **Jose Altuve** and **David Ortiz**
- Add more features: pitch count, inning, pitch type
- Deploy with **Streamlit** or **Flask** for live predictions

---

## 📂 Project Structure

```
ml-strikezone-classifier/
├── notebooks/
│   └── strikezone_model.ipynb
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🙋‍♀️ Author

**Arshita Sharma**  
Machine Learning Engineer Portfolio Project  
[GitHub](https://github.com/arshita08)
```

---
