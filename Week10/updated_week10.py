import pandas as pd
import streamlit as st
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import altair as alt
from sklearn.decomposition import PCA

# Fungsi untuk menghitung kata teratas dari seluruh data
def get_top_words(data, n=10):
    text = " ".join(data)
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    top_words = word_counts.most_common(n)
    return top_words

# Fungsi untuk membuat WordCloud dari seluruh data
def generate_wordcloud(data):
    text = " ".join(review for review in data)
    wordcloud = WordCloud(width=800, height=400, background_color="black").generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return plt

def main():
    # CSS untuk mengubah warna latar belakang
    st.markdown(
        """
        <style>
        /* Ubah warna background aplikasi */
        .stApp {
            background-color: #0C0C0C;
        }

        /* Custom warna untuk widget file uploader */
        .css-1cpxqw2, .css-qbe2hs, .css-1b7de37 {
            background-color: #FFFFF;
            border: 1px solid #B43F3F;
        }

        /* Ubah warna teks */
        .css-18e3th9 {
            color: #FFFFF;
        }

        /* Tambahkan padding untuk halaman */
        .block-container {
            padding-top: 12rem;
            padding-bottom: 4rem;
            padding-left: 20rem;
            padding-right: 20rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Membuat dua kolom: kolom kiri untuk judul, kolom kanan untuk GIF
    col1, col2 = st.columns([3, 4])  # Mengatur lebar kolom (3 bagian untuk kiri, 1 bagian untuk kanan)

    with col1:
        # Menampilkan judul dan deskripsi di kolom kiri
        st.markdown("")
        st.title("Visualisasi Komentar Youtube dan Sentimen")
        st.markdown("""
        Selamat datang di aplikasi visualisasi komentar Youtube dan sentimen! 
        Silakan unggah file CSV yang berisi data komentar dan sentiment untuk memulai visualisasi.
        """)

    with col2:
        # Menambahkan GIF di kolom kanan
        st.image("./gif/store.gif", use_column_width=True)

    # Membuat divider visual
    st.divider()

    # Upload file CSV di bawah kolom

    uploaded_file = st.file_uploader('Upload file CSV yang memiliki kolom "Comments" dan "Sentiment"', type=["csv"])

    if uploaded_file is not None:
        comments_df = pd.read_csv(uploaded_file)
        
        # Pastikan kolom "Comments" dan "Sentiment" ada
        if 'Comments' in comments_df.columns and 'Sentiment' in comments_df.columns:
            comments_df = comments_df.dropna(subset=['Comments', 'Sentiment'])

            # Step 1: Preprocessing text (Vectorization menggunakan TF-IDF)
            tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
            tfidf_matrix = tfidf_vectorizer.fit_transform(comments_df['Comments'])

            # Step 2: Melakukan Clustering dengan K-Means
            num_clusters = 3
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(tfidf_matrix)

            # Step 3: Menambahkan label cluster ke DataFrame
            comments_df['Cluster'] = kmeans.labels_

            # Dapatkan kata-kata teratas untuk seluruh data
            top_words = get_top_words(comments_df['Comments'], n=10)

            # Mempersiapkan DataFrame untuk diagram batang
            top_words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])

            # Buat diagram batang vertikal dengan Altair
            bar_chart = alt.Chart(top_words_df).mark_bar().encode(
                x=alt.X('Word:N', title='Words', sort='-y'),  # Mengatur orientasi vertikal
                y=alt.Y('Count:Q', title='Frequency'),
                color='Word:N'
            ).properties(
                title='Kata yang paling banyak muncul dalam komentar'
            )

            # Tampilkan diagram batang di Streamlit
            st.altair_chart(bar_chart, use_container_width=True)

            st.divider()
            # Menampilkan satu WordCloud untuk seluruh komentar
            st.subheader("Visualisasi WordCloud")
            st.markdown("Komentar Youtube")
            plt.figure(figsize=(10, 5))
            wordcloud = generate_wordcloud(comments_df['Comments'])
            st.pyplot(wordcloud)

            # Mengurangi dimensionalitas ke 2D untuk visualisasi (PCA)
            pca = PCA(n_components=2)
            reduced_tfidf = pca.fit_transform(tfidf_matrix.toarray())

            st.divider()
            st.subheader('Hasil Clustering')
            # Menampilkan hasil clustering dalam bentuk scatter plot
            df_viz = pd.DataFrame(reduced_tfidf, columns=['x', 'y'])
            df_viz['Cluster'] = comments_df['Cluster']

            scatter = alt.Chart(df_viz).mark_circle(size=60).encode(
                x='x',
                y='y',
                color='Cluster:N',
                tooltip=['x', 'y', 'Cluster']
            ).interactive().properties(
                title='Komentar Youtube'
            )

            st.altair_chart(scatter, use_container_width=True)

            st.divider()
            # Menampilkan 5 contoh pertama dari hasil clustering
            st.subheader("Contoh Data Komentar dan Sentimen")
            st.write("Gulir untuk melihat lebih banyak.")

            # Menampilkan DataFrame yang bisa di-scroll per 10 baris
            st.dataframe(comments_df[['Comments', 'Sentiment', 'Cluster']].head(200), height=300)

            st.divider()
            st.subheader("Visualisasi Distribusi Sentimen")
            sentiment_counts = comments_df['Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']

            # Buat diagram batang distribusi sentimen
            sentiment_chart = alt.Chart(sentiment_counts).mark_bar().encode(
                x=alt.X('Sentiment:N', title='Sentiment'),
                y=alt.Y('Count:Q', title='Count'),
                color='Sentiment:N'
            ).properties(
                title='Komentar Youtube'
            )
            
            st.altair_chart(sentiment_chart, use_container_width=True)

        else:
            st.error('File CSV harus memiliki kolom "Comments" dan "Sentiment".')

    else:
        st.info("Silakan unggah file CSV yang valid.")

if __name__ == '__main__':
    main()
