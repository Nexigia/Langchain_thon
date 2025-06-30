import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
# 개별 파일 로더들을 임포트합니다. UnstructuredPowerPointLoader 포함
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader 
# RecursiveCharacterTextSplitter만 사용합니다.
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import nltk 


# OpenAI API Key 설정
# Streamlit의 secrets에 'OPENAI_API_KEY'를 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ====================================
# PRE-PROCESSING 단계
# ====================================

class DocumentProcessor:
    """문서 전처리를 담당하는 클래스"""

    @staticmethod
    @st.cache_resource
    def load_documents(directory_path: str):
        """
        1. 문서 로드 (Document Load)
        (이 함수는 이제 직접적으로 사용되지 않습니다. 개별 파일 로딩 로직으로 대체됩니다.)
        """
        # st.warning("DocumentProcessor.load_documents는 현재 사용되지 않습니다. initialize_rag_system을 확인하세요.")
        return []

    @staticmethod
    def split_text(documents, chunk_size=100, chunk_overlap=20): 
        """
        2. Text Split (청크 분할)
        - 불러온 문서를 chunk 단위로 분할합니다.
        - RecursiveCharacterTextSplitter를 사용하여 글자 단위로 분할하며, OpenAI 토큰 제한을 위해 chunk_size를 매우 보수적으로 설정합니다.
        """
        text_splitter = RecursiveCharacterTextSplitter( 
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""] 
        )
        split_docs = text_splitter.split_documents(documents)
        # st.info(f"✂️ 텍스트 분할 완료: {len(split_docs)} 청크")
        return split_docs

    @staticmethod
    # @st.cache_resource # 이 캐시는 계속 주석 처리되어 있어야 합니다!
    def create_vector_store(_split_docs, embeddings): 
        """
        4. DB 저장 (Vector Store)
        - 변환된 벡터를 FAISS DB에 저장합니다.
        """
        vectorstore = FAISS.from_documents(_split_docs, embeddings)
        # st.success("💾 벡터 DB 생성 완료!")
        return vectorstore
    
    @staticmethod
    def add_documents_to_vector_store(vectorstore, split_docs, embeddings): 
        """
        기존 벡터 저장소에 새로운 문서 청크들을 추가합니다.
        """
        vectorstore.add_documents(split_docs) 
        # st.success("💾 벡터 DB에 문서 청크 추가 완료!")
        return vectorstore

# ====================================
# RUNTIME 단계
# ====================================

class RAGRetriever:
    """검색기(Retriever) 관리 클래스"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def get_retriever(self, search_type="similarity", k=5):
        """
        1. 검색 (Retrieve)
        - Vector DB에서 관련 문서를 찾는 검색기를 생성합니다. (k값을 5로 조정하여 더 많은 문맥 참조)
        """
        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        return retriever

class PromptManager:
    """프롬프트 관리 클래스"""

    @staticmethod
    def get_contextualize_prompt():
        """
        2. 프롬프트 (Prompt) - 대화 맥락화
        - 채팅 기록을 바탕으로 후속 질문을 독립적인 질문으로 재구성합니다.
        """
        contextualize_q_system_prompt = """주어진 채팅 히스토리와 최신 사용자 질문을 바탕으로,
        채팅 히스토리 없이도 이해할 수 있는 독립적인 질문으로 재구성하세요.
        질문에 답하지 말고, 필요시 재구성하거나 그대로 반환하세요."""

        return ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])

    @staticmethod
    def get_qa_prompt():
        """
        2. 프롬프트 (Prompt) - 질문 답변
        - 검색된 문맥을 바탕으로 답변을 생성하기 위한 프롬프트입니다.
        """
        qa_system_prompt = """당신은 주어진 문서를 기반으로 질문에 답변하는 AI 어시스턴트입니다.
        제공된 검색 결과를 바탕으로 질문에 답변하세요.

        답변 규칙:
        - 정확한 정보만 제공하세요.
        - 모르는 내용에 대해서는 '문서에 관련 정보가 없습니다.'라고 솔직하게 답변하세요.
        - 한국어로 친절하고 상세하게 답변하세요.
        - 가능하다면 관련된 문서의 출처(source)와 페이지(page)를 함께 언급해주세요.

        검색된 문맥:
        {context}"""

        return ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])
    
    @staticmethod
    def get_general_qa_prompt():
        """
        문서 검색 없이 일반적인 질문에 답변하기 위한 프롬프트입니다.
        """
        general_system_prompt = """당신은 유용한 AI 어시스턴트입니다. 사용자의 질문에 간결하고 정확하게 답변하세요.
        어떤 상황에서도 문서 검색을 시도하지 말고, 오직 당신의 일반 지식으로만 답변하세요."""
        return ChatPromptTemplate.from_messages([
            ("system", general_system_prompt),
            MessagesPlaceholder("history"), 
            ("human", "{input}"),
        ])

    @staticmethod
    def get_intent_detection_prompt(): # ★★★ 새로운 의도 감지 프롬프트 추가 ★★★
        """
        사용자 질문의 의도를 감지하기 위한 프롬프트입니다.
        """
        intent_system_prompt = """당신은 질문 분류 전문가입니다. 사용자의 질문이 다음 두 가지 중 어디에 해당하는지 분류하세요:
        - DOCUMENTS: 질문이 업로드된 문서의 내용과 관련된 것 같습니다. (정보 검색 필요)
        - GENERAL: 질문이 일반적인 지식에 관한 것이고, 문서 검색이 필요하지 않습니다.

        오직 'DOCUMENTS' 또는 'GENERAL' 중 하나로만 답변하세요. 다른 설명은 필요 없습니다.
        """
        return ChatPromptTemplate.from_messages([
            ("system", intent_system_prompt),
            ("human", "질문: {question}"),
        ])


class LLMManager:
    """
    3. 언어 모델 (LLM) 관리
    - GPT-4o-mini 등 다양한 모델을 선택할 수 있습니다.
    """

    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
        )

    def get_llm(self):
        return self.llm

class RAGChain:
    """RAG 체인 구성 및 관리"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.chain = self._build_chain()

    def _build_chain(self):
        prompt_manager = PromptManager()
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, prompt_manager.get_contextualize_prompt()
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt_manager.get_qa_prompt())
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain

    def get_conversational_chain(self, chat_history):
        return RunnableWithMessageHistory(
            self.chain,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )

# ====================================
# 메인 애플리케이션
# ====================================

@st.cache_resource 
def download_nltk_data():
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    nltk.data.path.append(nltk_data_path)

    datasets = ['punkt', 'averaged_perceptron_tagger']

    for dataset in datasets:
        try:
            if dataset == 'punkt':
                nltk.data.find(f'tokenizers/{dataset}')
            else: 
                nltk.data.find(f'taggers/{dataset}')
        except LookupError: 
            try:
                nltk.download(dataset, quiet=True, download_dir=nltk_data_path)
            except Exception as e_download: 
                st.error(f"NLTK '{dataset}' 데이터 다운로드 최종 실패: {e_download}") 
                return None 
        except Exception as e_other: 
            st.error(f"NLTK '{dataset}' 데이터 확인 중 예상치 못한 오류 발생: {e_other}") 
            return None 
    return True 


def initialize_rag_system(model_name):
    """RAG 시스템 초기화 (개별 문서 처리 방식)"""
    # st.info("🔄 RAG 시스템 초기화 중...") 
    
    data_path = "./data" # 문서 폴더 경로
    vectorstore = None # 초기 벡터 저장소는 None으로 설정
    
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small') 
    
    # st.info("📂 문서 폴더에서 파일을 찾고 있습니다...") 

    processed_any_document = False
    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)
        
        if os.path.isfile(filepath): 
            # st.info(f"📄 파일 로드 시작: {filename}") 
            try:
                if filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                elif filename.lower().endswith(".docx"):
                    loader = Docx2txtLoader(filepath)
                elif filename.lower().endswith(".pptx"):
                    loader = UnstructuredPowerPointLoader(filepath) 
                elif filename.lower().endswith(".txt"):
                    loader = TextLoader(filepath)
                # elif filename.lower().endswith(".csv"): # CSV 파일 처리가 필요하다면 이 주석을 해제하고 CSVLoader를 임포트 및 requirements.txt에 pandas 추가
                #     loader = CSVLoader(filepath)
                else:
                    # st.warning(f"지원하지 않는 파일 형식입니다: {filename}. 건너킵니다.") 
                    continue 

                single_document_list = loader.load() 
                
                if not single_document_list:
                    # st.warning(f"파일 {filename}에서 문서를 로드하지 못했습니다. 건너킵니다.") 
                    continue

                split_single_doc_chunks = DocumentProcessor.split_text(single_document_list)
                
                if vectorstore is None:
                    vectorstore = DocumentProcessor.create_vector_store(split_single_doc_chunks, embeddings)
                else:
                    vectorstore = DocumentProcessor.add_documents_to_vector_store(vectorstore, split_single_doc_chunks, embeddings)
                
                processed_any_document = True

            except Exception as e:
                st.error(f"❌ 파일 {filename} 처리 중 오류 발생: {e}") 
                continue 

    if not processed_any_document or vectorstore is None:
        st.error("❌ 'data' 폴더에 처리할 문서가 없거나 모든 문서 처리 중 오류가 발생하여 벡터 DB를 생성하지 못했습니다.") 
        return None 

    # st.success("✅ 모든 문서 처리 및 벡터 DB 생성 완료!") 
    
    # 벡터 저장소가 성공적으로 생성된 경우 RAG 체인 구성
    rag_retriever = RAGRetriever(vectorstore)
    retriever = rag_retriever.get_retriever()
    llm_manager = LLMManager(model_name)
    llm = llm_manager.get_llm()
    rag_chain = RAGChain(retriever, llm) # RAG 체인
    
    # st.success("✅ RAG 시스템 초기화 완료!") 
    return rag_chain, llm # rag_chain과 LLM 인스턴스를 함께 반환 (일반 질문 체인에 사용)


def format_output(response):
    """결과 포맷팅"""
    if isinstance(response, dict):
        answer = response.get('answer', '답변을 생성할 수 없습니다.')
        context = response.get('context', [])
    else: 
        answer = response # LLM이 직접 문자열을 반환하는 경우
        context = []

    return {
        'answer': answer,
        'context': context,
        'source_count': len(context)
    }

# ====================================
# Streamlit UI
# ====================================

def main():

    # NLTK 데이터 다운로드
    nltk_download_status = download_nltk_data()
    if nltk_download_status is None: 
        return

    st.set_page_config(
        page_title="RAG 문서 Q&A 챗봇",
        page_icon="🤖",
        layout="wide"
    )

    st.header("🤖 RAG 기반 문서 Q&A 챗봇 💬")
    st.markdown("`data` 폴더의 문서(PDF, TXT, DOCX 등)를 기반으로 질문에 답변합니다.")

    with st.sidebar:
        st.header("🔧 설정")
        model_option = st.selectbox(
            "GPT 모델 선택",
            ("gpt-4o-mini", "gpt-3.5-turbo-0125", "gpt-4o"),
            help="사용할 GPT 모델을 선택하세요"
        )
        st.markdown("---")
        # is_general_question_mode 체크박스 제거
        st.info("`data` 폴더에 파일을 추가/삭제한 후에는 페이지를 새로고침하여 시스템을 다시 초기화해주세요.")
        st.markdown("---")
        st.markdown("### 📊 RAG 프로세스")
        st.markdown("""
        **Pre-processing:**
        1. 📄 문서 로드 (개별 파일 처리)
        2. ✂️ 텍스트 분할 (매우 작은 청크)
        3. 💾 벡터 DB 저장/추가 (각 문서 청크별)

        **Runtime (자동 라우팅):**
        1. 🤔 질문 의도 감지 (문서 관련 vs 일반 지식)
        2. 🔍 유사도 검색 (필요시)
        3. 📝 프롬프트 구성
        4. 🤖 LLM 추론
        5. 📋 결과 출력
        """)

    # initialize_rag_system은 이제 rag_chain과 llm(general_llm)을 함께 반환합니다.
    rag_chain_wrapper, llm_for_general_qa = initialize_rag_system(model_option) # 이름 변경
    
    if rag_chain_wrapper is None: # RAG 시스템 초기화 실패 시 앱 중단
        return

    chat_history = StreamlitChatMessageHistory(key="chat_messages")
    conversational_rag_chain = rag_chain_wrapper.get_conversational_chain(chat_history)
    
    # --- 일반 질문 모드용 LLM 체인 생성 ---
    prompt_manager = PromptManager() 
    general_llm_chain_template = prompt_manager.get_general_qa_prompt() 
    
    general_qa_chain_raw = create_stuff_documents_chain(llm_for_general_qa, general_llm_chain_template) 
    general_conversational_chain = RunnableWithMessageHistory(
        general_qa_chain_raw, 
        lambda session_id: chat_history, 
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer", 
    )
    # --- 의도 감지 체인 생성 ---
    intent_detection_prompt = prompt_manager.get_intent_detection_prompt()
    # 이 체인은 대화 히스토리 없이 사용자의 질문만 받도록 간단하게 구성
    intent_detection_chain = ChatOpenAI(model=model_option, temperature=0).bind(
        response_format={"type": "text"} # 텍스트 형식으로 응답 받도록 강제
    ).with_structured_output(
        schema={"type": "string", "enum": ["DOCUMENTS", "GENERAL"]} # 응답을 특정 문자열로 제한
    ) # ★★★ 의도 감지 체인 생성 시 형식 지정 ★★★
    
    # 의도 감지 체인을 RunnableWithMessageHistory로 래핑하여 채팅 히스토리도 사용 가능하게 함 (선택 사항)
    # 그러나 의도 감지는 보통 질문 자체에 집중하므로 간단한 체인으로도 충분.
    # 여기서는 간단하게 만들고, 호출 시 chat_history는 무시.


    # 초기 메시지를 일반적인 내용으로 변경
    if not chat_history.messages:
        chat_history.add_ai_message("안녕하세요! `data` 폴더의 문서에 대해 무엇이든 물어보세요! 📚")

    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input("문서에 대해 궁금한 점을 입력해주세요..."):
        st.chat_message("human").write(prompt)

        with st.chat_message("ai"):
            with st.spinner("🧐 질문을 분석하고 답변을 생성 중입니다..."): 
                try:
                    # 1. 질문 의도 감지
                    # 의도 감지 시에는 채팅 히스토리 없이 현재 질문만 보내는 것이 더 명확할 수 있습니다.
                    intent_response = intent_detection_chain.invoke(
                        {"question": prompt}
                    )
                    
                    # LLM의 응답이 dict 형태라면 'text' 키를, 아니면 직접 문자열을 사용
                    # structured_output을 사용했으므로 바로 문자열이 올 것으로 예상
                    intent = intent_response # 바로 'DOCUMENTS' 또는 'GENERAL' 문자열

                    if intent.strip().upper() == "GENERAL": # LLM이 반환한 의도에 따라 분기
                        st.info("💡 일반적인 질문으로 판단하여 LLM의 일반 지식으로 답변합니다.")
                        config = {"configurable": {"session_id": "general_chat"}}
                        response = general_conversational_chain.invoke({"input": prompt}, config)
                        st.write(response['answer']) # response는 dict 형태일 것임 (answer 키 포함)
                        # 일반 질문 모드이므로 참고 문서는 표시하지 않음

                    elif intent.strip().upper() == "DOCUMENTS":
                        st.info("🔍 문서 관련 질문으로 판단하여 문서 검색 후 답변합니다.")
                        config = {"configurable": {"session_id": "rag_chat"}}
                        response = conversational_rag_chain.invoke({"input": prompt}, config)
                        formatted_response = format_output(response)
                        st.write(formatted_response['answer'])

                        with st.expander(f"📄 참고 문서 ({formatted_response['source_count']}개)"):
                            if formatted_response['context']:
                                for i, doc in enumerate(formatted_response['context']):
                                    st.markdown(f"**📖 문서 {i+1}**")
                                    source = doc.metadata.get('source', '출처 정보 없음')
                                    st.markdown(f"**출처:** `{source}`")
                                    if 'page' in doc.metadata:
                                        page = doc.metadata.get('page')
                                        st.markdown(f"**페이지:** {page + 1}")
                                    st.text_area(
                                        f"문서 {i+1} 내용",
                                        doc.page_content,
                                        height=150,
                                        key=f"doc_{i}",
                                        label_visibility="collapsed"
                                    )
                                    if i < len(formatted_response['context']) - 1:
                                        st.markdown("---")
                            else:
                                st.info("답변에 참고한 문서를 찾을 수 없습니다.")
                    else:
                        st.warning(f"의도 파악에 실패했습니다. (응답: {intent}). 기본적으로 RAG 모드로 진행합니다.")
                        config = {"configurable": {"session_id": "rag_chat"}}
                        response = conversational_rag_chain.invoke({"input": prompt}, config)
                        formatted_response = format_output(response)
                        st.write(formatted_response['answer'])

                        with st.expander(f"📄 참고 문서 ({formatted_response['source_count']}개)"):
                            if formatted_response['context']:
                                for i, doc in enumerate(formatted_response['context']):
                                    st.markdown(f"**📖 문서 {i+1}**")
                                    source = doc.metadata.get('source', '출처 정보 없음')
                                    st.markdown(f"**출처:** `{source}`")
                                    if 'page' in doc.metadata:
                                        page = doc.metadata.get('page')
                                        st.markdown(f"**페이지:** {page + 1}")
                                    st.text_area(
                                        f"문서 {i+1} 내용",
                                        doc.page_content,
                                        height=150,
                                        key=f"doc_{i}",
                                        label_visibility="collapsed"
                                    )
                                    if i < len(formatted_response['context']) - 1:
                                        st.markdown("---")
                            else:
                                st.info("답변에 참고한 문서를 찾을 수 없습니다.")

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")
                    st.info("다시 시도해주세요.")

if __name__ == "__main__":
    main()
