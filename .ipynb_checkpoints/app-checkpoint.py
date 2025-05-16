import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client
import random


# 📥 Veriyi yükle
@st.cache_data
def load_data():
    df = pd.read_excel("Mentoring_data.xlsx")
    df.dropna(inplace=True)
    return df

# 🔍 Benzer cevabı bul
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_answer(user_question, questions, answers, threshold=0.5):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([user_question] + questions)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    max_sim_index = similarities.argmax()
    max_sim_value = similarities[max_sim_index]
    if max_sim_value > threshold:
        return answers[max_sim_index]
    else:
        return "ℹ️ Sorunuz, bu chatbotun bilgi alanı dışında olabilir. Chatbot yalnızca veri bilimi ve ilgili teknik konulara odaklanır. Daha detaylı yardım için mentorunuza danışabilirsiniz." 

def main():
    st.set_page_config(page_title="DataMentor", page_icon="🌱")
     # 🔵 LOGO ALANI (buraya şirket logosu eklenecek)
    st.sidebar.image("logo.svg", use_container_width=True)  # <-- LOGOYU BURAYA KOY ✅
    st.sidebar.title("🔧 Menü")
    page = st.sidebar.selectbox("Sayfa Seçin", ["🧠 Chatbot", "📚 Kaynak Önerileri", "ℹ️ Hakkında", "📞 İletişim"])

    if page == "📚 Kaynak Önerileri":
        st.title("📚 Data Science Kaynak Önerileri")
        st.markdown("İlgilendiğin konu başlığına tıklayarak önerilen ücretsiz kaynakları görebilirsin:")

        konu_basliklari = {
            "Python": [
                "📘 [W3Schools Python Tutorial](https://www.w3schools.com/python/)",
                "📘 [Python.org Resmi Belgeler](https://docs.python.org/3/)",
                "🧑‍💻 [Programiz Python Tutorial (TR destekli)](https://www.programiz.com/python-programming)"
            ],
            "Pandas": [
                "📘 [Pandas Resmi Belgeler](https://pandas.pydata.org/docs/)",
                "🧑‍💻 [Kaggle Learn: Pandas](https://www.kaggle.com/learn/pandas)",
                "📘 [GeeksforGeeks Pandas Tutorials](https://www.geeksforgeeks.org/python-pandas-tutorial/)"
            ],
            "Makine Öğrenmesi (ML)": [
                "📘 [Scikit-learn Tutorial](https://scikit-learn.org/stable/tutorial/index.html)",
                "🧑‍💻 [Kaggle Learn: ML](https://www.kaggle.com/learn/intro-to-machine-learning)",
                "📘 [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)"
            ],
            "Derin Öğrenme (DL)": [
                "📘 [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)",
                "📘 [PyTorch Tutorials](https://pytorch.org/tutorials/)",
                "🧑‍💻 [DeepLizard YouTube](https://www.youtube.com/playlist?list=PLZyvi_9gamL-EE3zQJbU5N5o8lZ1msDJM)"
            ],
            "Veri Görselleştirme": [
                "📘 [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)",
                "📘 [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)",
                "🧑‍💻 [Kaggle Learn: Data Visualization](https://www.kaggle.com/learn/data-visualization)"
            ],
            "Veri Ön İşleme": [
                "📘 [Scikit-learn: Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)",
                "📘 [GeeksforGeeks: Preprocessing](https://www.geeksforgeeks.org/data-preprocessing-machine-learning-python/)",
                "🧑‍💻 [Kaggle Learn: Feature Engineering](https://www.kaggle.com/learn/feature-engineering)"
            ],
            "📚 Kitaplar": [
                "📘 [Automate the Boring Stuff (Ücretsiz)](https://automatetheboringstuff.com)",
                "📘 [Machine Learning Yearning (Andrew Ng - Ücretsiz PDF)](https://www.deeplearning.ai/machine-learning-yearning/)",
                "📘 [Think Stats (Ücretsiz)](http://greenteapress.com/wp/think-stats/)",
                "📘 [An Introduction to Statistical Learning (Ücretsiz PDF)](https://www.statlearning.com/)",
                "📘 [Speech and Language Processing (Jurafsky)](https://web.stanford.edu/~jurafsky/slp3/)"
            ],
            "🇹🇷 Türkçe Kaynaklar": [
                "📝 [Veri Bilimi Türkiye - Medium](https://medium.com/veri-bilimi-t%C3%BCrkiye)",
                "🎥 [BTK Akademi - Veri Bilimi Eğitimleri](https://btkakademi.gov.tr/)",
                "🧑‍🏫 [Patika.dev - Python & ML](https://www.patika.dev/)"
            ]
        }

        for konu, kaynaklar in konu_basliklari.items():
            with st.expander(f"🔹 {konu}"):
                for kaynak in kaynaklar:
                    st.markdown(f"- {kaynak}")
        return


            # ℹ️ Hakkında Sayfası
    elif page == "ℹ️ Hakkında":
        st.title("ℹ️ Hakkında")
        st.info("Bu chatbot, TechPro Education'ın Data Science öğrencileri için geliştirilmiş bir sanal mentordur. \
        Öğrencilerin sıkça karşılaştığı sorunlara yapay zekâ destekli hızlı cevaplar sunar. Model, yalnızca veri bilimi, yapay zekâ ve yazılım alanlarına odaklanmıştır.")
        st.markdown("""
        ### 🎯 Amaç:
        - Öğrencilerin kendi başlarına sorunlarını çözmelerine yardımcı olmak
        - Mentorluk yükünü azaltmak
        - Anında destek sağlayarak öğrenme sürecini hızlandırmak

        ### 🛠️ Odak Konular:
        - Veri Bilimi Temelleri
        - Makine Öğrenmesi Algoritmaları
        - Python ve Kütüphaneleri
        - Proje Geliştirme ve Kaggle Yarışmaları
        - Motivasyon ve Zaman Yönetimi
        """)

    elif page == "📞 İletişim":
        st.title("📞 Bizimle İletişime Geçin")
        st.markdown("""
        ### 📲 Sosyal Medya & İletişim Bilgileri

        - 📞 **Telefon:** +1 (587) 330-7959  
        - 📷 [**Instagram**](https://www.instagram.com/techproeducationtr/)  
        - 📻 [**YouTube**](https://www.youtube.com/@TechProEducation)  
        - 💼 [**LinkedIn**](https://www.linkedin.com/school/techproeducationtr/)  
        - 🌐 [**Web Sitesi**](https://www.techproeducation.com.tr/)
        """)
        return

    elif page == "🧠 Chatbot":
        st.title("🌱 DataMentor")
        st.write("Aklına takılan soruları cevaplamak için buradayım! Ancak, zaman zaman hata yapabileceğimi unutma. Önemli konular için  mentoruna  danışman her zaman daha doğru olacaktır.")

        df = load_data()
        questions = df["questions"].tolist()
        answers = df["answers"].tolist()

        st.markdown("### 💡 Popüler Konular")
        popular_questions = [
            "Makine öğrenmesi nedir?",
            "Linkedin profilimde neler olmalı?",
            "Veri görselleştirme araçları nelerdir?",
            "Motivasyona ihtiyacim var"
        ]
        selected_popular = st.columns(len(popular_questions))
        for i, q in enumerate(popular_questions):
            if selected_popular[i].button(q):
                st.session_state.user_input = q

        user_input = st.text_input("Sorunuz:", key="user_input")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # 🎯 Motivasyon Cümleleri
        motivation_quotes = [
    "🚀 Küçük adımlar büyük değişimlere yol açar. Bugün attığın minicik bir adım, yarın seni hayal bile edemeyeceğin bir noktaya taşıyabilir. Vazgeçme, çünkü başarı sabırla büyür.",
    
    "🌟 En karanlık an, yıldızların en parlak şekilde parladığı andır. Zor zamanlar seni pes ettirmek için değil, içindeki gücü fark etmen için vardır. Işığını kaybetme.",
    
    "💡 Hedeflerine ulaşmak zaman alabilir ama hiçbir zaman imkânsız değildir. Her gün attığın adım seni biraz daha yaklaştırır. Sabırlı ol, çünkü zaman senin en güçlü müttefikin olabilir.",
    
    "🔥 Başarısızlık, sadece pes ettiğin an olur. Denemeye devam ettiğin sürece kaybetmiş sayılmazsın. Unutma, her deneme seni başarıya bir adım daha yaklaştırır.",
    
    "🏁 Bugünün zorluğu, yarının gücünü oluşturur. Direndiğin her an, seni daha dayanıklı, daha bilgili ve daha kararlı biri yapar. Bu yolculuk seni şekillendiriyor.",
    
    "📈 Her deneme seni başarıya bir adım daha yaklaştırır. Mükemmelliğe ulaşmak, bir anda değil; deneme, yanılma ve öğrenme süreciyle olur. Hata yapmaktan korkma, öğrenmek cesaret ister.",
    
    "✨ Başkalarının imkânsız dediği şey, senin için sadece henüz yapılmamış bir başarıdır. Sınırları başkaları değil, sen belirlersin. Kendi hikayeni yazmaktan korkma.",
    
    "🛠️ Her hata bir öğrenme fırsatıdır. Unutma, en iyi öğrenmeler başarısızlıkların içinden doğar. Hatalar seni tanımlamaz; onlara verdiğin tepkiler seni güçlü kılar.",
    
    "🎯 İlerlemek zor olabilir ama yerinde saymak daha da zor. Çünkü hayallerin seni çağırıyor ve sen bu yolculuğa çıkabilecek kadar cesur birisin. Vazgeçmek yok!",
    
    "🌱 Gelişim, rahat alanın dışında başlar. Konfor alanının ötesine geçtiğinde, gerçek potansiyelini keşfedersin. Korkma, çünkü büyümek cesaret ister."
]

        if user_input:
            trigger_words = ["motivasyon", "moral", "üzgünüm", "ihtiyacım var", "bunalım", "motive", "depresyondayım"]

            if any(word in user_input.lower() for word in trigger_words):
                answer = random.choice(motivation_quotes)
            else:
                answer = get_answer(user_input, questions, answers)

            st.session_state.chat_history.append((user_input, answer))

            st.markdown("---")
            st.markdown("### 🗨️ Sohbet Geçmişi")
            
            # 🗑️ Sohbet geçmişini temizleme butonu
            if st.button("🗑️ Geçmişi Temizle"):
                st.session_state.chat_history = []
            
            # 📋 Sohbet geçmişini göster
            for idx, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                st.markdown(f"**❓ Soru {idx}:** {q}")
                st.markdown(f"**✅ Cevap:** {a}")
                st.markdown("---")

     

if __name__ == "__main__":
    main()