import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import altair as alt

# Load dataset yang sudah di-clean dan memiliki hasil sentimen
comments_df = pd.read_csv('./dataset/comments_sentiment.csv')  # Pastikan file ini memiliki kolom "Comments"
comments_df = comments_df.dropna(subset=['Comments'])

# Step 1: Preprocessing text (Vectorization menggunakan TF-IDF)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)  # Mengabaikan kata yang terlalu sering muncul
tfidf_matrix = tfidf_vectorizer.fit_transform(comments_df['Comments'])  # Representasi teks dalam bentuk vektor

# Step 2: Melakukan Clustering dengan K-Means
num_clusters = 3  # Tentukan jumlah cluster yang diinginkan
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

# Step 3: Menambahkan label cluster ke DataFrame
comments_df['Cluster'] = kmeans.labels_

# Step 4: Visualisasi hasil clustering
def plot_clusters(data, labels, title="K-Means Clustering"):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')
    plt.title(title)
    plt.show()

# Mengurangi dimensionalitas ke 2D untuk visualisasi (menggunakan PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_tfidf = pca.fit_transform(tfidf_matrix.toarray())

# Menampilkan hasil clustering dalam bentuk scatter plot
plot_clusters(reduced_tfidf, kmeans.labels_, title="K-Means Clustering with 3 Clusters")

# Step 5: Visualisasi dengan Altair
st.title("Clustering Hasil Analisis Sentimen")
st.subheader("Visualisasi Clustering dengan K-Means")
st.divider()

# Membuat dataframe untuk visualisasi dengan Altair
df_viz = pd.DataFrame(reduced_tfidf, columns=['x', 'y'])
df_viz['Cluster'] = comments_df['Cluster']

# Visualisasi dengan Altair
scatter = alt.Chart(df_viz).mark_circle(size=60).encode(
    x='x',
    y='y',
    color='Cluster:N',  # Cluster sebagai kategori
    tooltip=['x', 'y', 'Cluster']
).interactive()

st.altair_chart(scatter, use_container_width=True)

# Menampilkan 5 baris pertama dari hasil clustering
st.write(comments_df[['Comments', 'Cluster']].head(10))

def main():
    st.title("Hasil Analisis Sentimen Bahasa Indonesia")
    st.subheader("Hasil Analisis Sentimen")
    st.divider()
    #st.image("Studying-amico.png", use_column_width=True)

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":

        # Upload file CSV
        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

        if uploaded_file is not None:
            # Membaca file CSV
            results_df = pd.read_csv(uploaded_file)

            # Pastikan kolom yang diperlukan ada
            if 'Comments' in results_df.columns and 'Sentiment' in results_df.columns:
                # Layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("Hasil Analisis Sentimen")
                    for index, row in results_df.head(5).iterrows():
                        st.write(f"Komentar: {row['Comments']}")
                        st.write(f"Sentimen: {row['Sentiment']}")

                        # Emoji berdasarkan hasil sentimen
                        if row['Sentiment'] == 'positif':
                            st.markdown("Sentiment: Positive 😀")
                        else:
                            st.markdown("Sentiment: Negative 😭")

                with col2:
                    # Menampilkan DataFrame hasil analisis
                    st.dataframe(results_df)

                    # Visualisasi sentimen
                    sentiment_counts = results_df['Sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']

                    c = alt.Chart(sentiment_counts).mark_bar().encode(
                        x='Sentiment',
                        y='Count',
                        color='Sentiment'
                    )
                    st.altair_chart(c, use_container_width=True)
            else:
                st.error("File CSV tidak memiliki kolom yang diperlukan: 'Comments' dan 'Sentiment'.")
    
    else:
        st.subheader("About")

if __name__ == '__main__':
    main()

