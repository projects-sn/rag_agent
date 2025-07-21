import streamlit as st
from rag_agent import build_rag_chain

st.set_page_config(page_title="RAG по стенограммам", layout="wide")

st.title("📋 AI-агент по стенограммам GR-комитета")
st.markdown("Задайте вопрос по содержанию встреч — AI найдёт ответ по сводкам.")

# Построение цепочки
rag_chain = build_rag_chain()

# Кэшируем ответ и источники
@st.cache_data(show_spinner="🔎 Обрабатываем запрос...")
def get_rag_answer_and_sources(query: str):
    result = rag_chain(query)
    return result["result"], result.get("source_documents", [])

# Ввод вопроса
query = st.text_input(
    "Введите вопрос",
    placeholder="Что важного было обсуждено по образовательным инициативам?",
    key="user_question"
)

if query:
    answer, sources = get_rag_answer_and_sources(query)

    st.markdown("### 📌 Ответ:")
    st.write(answer)

    # Кнопка для показа источников
    if st.button("Показать источники"):
        if not sources:
            st.info("Источники не найдены.")
        else:
            st.markdown("### 📚 Использованные источники:")
            for i, doc in enumerate(sources):
                meta = doc.metadata
                date = meta.get("date", "неизвестно")
                doc_id = meta.get("doc_id", "N/A")
                st.markdown(f"**Источник {i+1}** — Документ №{doc_id}, дата: {date}")
                st.code(doc.page_content[:1000], language="markdown")  # выводим до 1000 символов
