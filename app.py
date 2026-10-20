import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. إعداد واجهة المستخدم
st.set_page_config(page_title="IT-Copilot | المساعد التقني للشركة", page_icon="💻", layout="centered")

st.title("💻 IT-Copilot: المساعد الذكي لموظفي الشركة")
st.caption("احصل على حلول فورية لمشاكل الشبكات، كلمات المرور، وإعدادات العمل عن بُعد.")

# 2. ملف معرفة تجريبي (Mock KB) لأغراض الـ Demo
DOC_CONTENT = """
1. إعدادات شبكة VPN للعمل عن بعد:
- قم بتنزيل تطبيق OpenVPN أو FortiClient من بوابة برامج الشركة.
- أدخل خادم الاتصال: vpn.company.com.
- اسم المستخدم هو بريدك الرسمي، وكلمة المرور هي نفس كلمة مرور الدخول للكمبيوتر متبوعة برمز التحقق الثنائي (2FA).
- في حال استمرار مشكلة الاتصال، تأكد من إيقاف جدار الحماية المنزلي مؤقتاً أو أعد تشغيل الراوتر.

2. إعادة تعيين كلمة المرور:
- يمكن تغيير كلمة المرور عبر الرابط الداخلي: https://portal.company.com/reset.
- يجب أن تحتوي كلمة المرور على 12 خانة على الأقل وتشمل رموزاً وأرقاماً وأحرفاً كبيرة.
- لا يمكن تكرار آخر 4 كلمات مرور استخدمتها سابقاً.

3. طلب أجهزة جديدة أو ملحقات:
- الدخول إلى قسم طلبات IT عبر بوابة الخدمة الذاتية.
- يتطلب اعتماد المدير المباشر لأي شاشة إضافية أو ترقية ذاكرة.
"""

# حفظ المستند محلياً للبدء
if not os.path.exists("kb.txt"):
    with open("kb.txt", "w", encoding="utf-8") as f:
        f.write(DOC_CONTENT)

# 3. إعداد محرك البحث والاسترجاع (RAG Pipeline)
@st.cache_resource
def setup_rag():
    loader = TextLoader("kb.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    
    # استخدام FakeEmbeddings للتجربة السريعة، أو بدّلها بـ OpenAIEmbeddings / HuggingFace
    embeddings = FakeEmbeddings(size=128)
    vectorstore = Chroma.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = setup_rag()

# 4. إدارة حالة المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "مرحباً بك! كيف يمكنني مساعدتك في المشاكل التقنية اليوم؟"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 5. التفاعل مع المستخدم
user_query = st.chat_input("اكتب استفسارك هنا (مثال: كيف أضبط الـ VPN؟)...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # استرجاع سياق المعرفة
    matched_docs = retriever.invoke(user_query)
    context_text = "\n\n".join([doc.page_content for doc in matched_docs])

    # صياغة الإجابة التوضيحية
    response = f"**بناءً على سياسات ودليل الـ IT الرسمي:**\n\n{context_text}\n\n---\n*إذا لم تحل المشكلة، يرجى التوجه لبوابة الدعم لفتح تذكرة مباشرة.*"

    with st.chat_message("assistant"):
        st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# 6. زر جانبي لفتح تذكرة سريعة
with st.sidebar:
    st.header("خدمات سريعة")
    if st.button("🚨 رفع تذكرة دعم للمستوى الثاني (L2)"):
        st.success("تم إرسال ملخص المشكلة لفريق الدعم وسيتواصل معك فني خلال 30 دقيقة.")
