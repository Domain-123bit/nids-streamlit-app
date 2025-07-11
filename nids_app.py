import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and tools using full paths
import gzip
with gzip.open('random_forest_nslkdd_compressed.pkl.gz', 'rb') as f:
    model = joblib.load(f)
scaler = joblib.load('scaler.pkl')
label_encoders = {}  # Set to empty if not used

# Define input feature columns
feature_columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                   'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
                   'num_failed_logins', 'logged_in', 'num_compromised', 
                   'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
                   'num_shells', 'num_access_files', 'num_outbound_cmds', 
                   'is_host_login', 'is_guest_login', 'count', 'srv_count', 
                   'serror_rate', 'srv_serror_rate', 'rerror_rate', 
                   'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
                   'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
                   'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
                   'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
                   'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
                   'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

# Streamlit UI
st.set_page_config(page_title="NIDS App", layout="wide")
st.title("🔐 AI-Based Network Intrusion Detection System")
st.markdown("Enter network features below to predict whether it's **Normal** or an **Attack**.")

user_input = []
for col in feature_columns:
    if col in label_encoders:
        options = label_encoders[col].classes_
        val = st.selectbox(f"{col}", options)
        val = label_encoders[col].transform([val])[0]
    else:
        val = st.number_input(f"{col}", step=1.0, format="%.3f")
    user_input.append(val)

if st.button("🛡️ Detect Intrusion"):
    input_array = np.array([user_input])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    label = "✅ Normal" if prediction[0] == 0 else "🚨 Attack"
    st.success(f"Prediction: {label}")

st.markdown("---")
st.subheader("📁 Or Upload a CSV File for Bulk Prediction")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)

    st.write("📋 Preview of Uploaded Data:")
    st.dataframe(df_uploaded.head())

    # Encode categorical columns if needed
    for col in df_uploaded.columns:
        if col in label_encoders:
            df_uploaded[col] = label_encoders[col].transform(df_uploaded[col])

    # Scale features (ignore 'target' if it's in test file)
    X_upload = df_uploaded.drop(columns=['target'], errors='ignore')
    X_upload_scaled = scaler.transform(X_upload)

    # Make predictions
    predictions = model.predict(X_upload_scaled)
    df_uploaded['Prediction'] = np.where(predictions == 0, 'Normal ✅', 'Attack 🚨')

    st.success("✅ Predictions Complete!")
    st.dataframe(df_uploaded)
    csv_download = df_uploaded.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results as CSV", data=csv_download, file_name="nids_results.csv", mime="text/csv")

st.markdown("---")
st.subheader("📈 Model Performance on NSL-KDD Test Set")

if st.button("🔍 Evaluate on Test Set"):
    try:
        # Load test set
        test_path = r'C:\KDDTest+.txt'
        col_path = r'C:\Field Names.csv'

        # Load columns
        col_names = pd.read_csv(col_path, header=None)[0].tolist()
        col_names.append('target')
        df_test = pd.read_csv(test_path, names=col_names)

        # Encode categorical columns
        for col in ['protocol_type', 'service', 'flag']:
            if col in df_test.columns and col in label_encoders:
                df_test[col] = label_encoders[col].transform(df_test[col])

        # Encode target using a fresh encoder
        from sklearn.preprocessing import LabelEncoder
        le_target = LabelEncoder()
        df_test['target'] = le_target.fit_transform(df_test['target'])

        # Scale and predict
        X_test = df_test.drop('target', axis=1)
        y_test = df_test['target']
        X_test_scaled = scaler.transform(X_test)

        y_pred = model.predict(X_test_scaled)

        # Show metrics
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Accuracy: {acc:.4f}")

        st.text("📊 Classification Report:")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
import matplotlib.pyplot as plt

st.markdown("---")
st.subheader("📊 Feature Importance (Random Forest)")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feature_names = [f'Feature {i}' for i in range(len(importances))]

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig, ax = plt.subplots()
    ax.barh(fi_df["Feature"], fi_df["Importance"], color='skyblue')
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 10 Important Features")
    st.pyplot(fig)
else:
    st.info("Feature importance not available for this model.")
