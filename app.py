import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
# 개별 파일 로더들을 임포트합니다. UnstructuredPowerPointLoader, UnstructuredFileLoader 포함
# PyPDFLoader 대신 UnstructuredFileLoader를 PDF에 사용합니다.
from langchain_community.document_loaders import Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredFileLoader 
# RecursiveCharacterTextSplitter만 사용합니다.
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 오타 수정
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# AIMessage 임포트 추가
from langchain_core.messages import AIMessage
import nltk 

# ★★★ st.set_page_config()를 main() 함수 밖으로 이동하여 앱 시작 시 한 번만 호출되도록 합니다. ★★★
st.set_page_config(
    page_title="RAG 문서 Q&A 챗봇",
    page_icon="🤖",
    layout="wide"
)

# OpenAI API Key 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ====================================
# PRE-PROCESSING 단계
# ====================================

class DocumentProcessor:
    """문서 전처리를 담당하는 클래스"""

    @staticmethod
    @st.cache_resource
    def load_documents(directory_path: str):
        # 이 함수는 현재 사용되지 않습니다. initialize_rag_system을 확인하세요.
        return []

    @staticmethod
    def split_text(documents, chunk_size=50, chunk_overlap=10):
        """
        2. 텍스트 분할 (청크 분할)
        - 불러온 문서를 청크 단위로 분할합니다.
        - RecursiveCharacterTextSplitter를 사용하여 글자 단위로 분할하며, OpenAI 토큰 제한을 위해 chunk_size를 매우 보수적으로 설정합니다.
        """
        text_splitter = RecursiveCharacterTextSplitter(  # 올바른 클래스명
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""] 
        )
        split_docs = text_splitter.split_documents(documents)
        return split_docs

    @staticmethod
    # @st.cache_resource 
    def create_vector_store(_split_docs, embeddings): 
        """
        4. DB 저장 (벡터 저장소)
        - 변환된 벡터를 FAISS DB에 저장합니다.
        """
        vectorstore = FAISS.from_documents(_split_docs, embeddings)
        return vectorstore
    
    @staticmethod
    def add_documents_to_vector_store(vectorstore, split_docs, embeddings): 
        """
        기존 벡터 저장소에 새로운 문서 청크들을 추가합니다.
        """
        vectorstore.add_documents(split_docs) 
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
        - 벡터 DB에서 관련 문서를 찾는 검색기를 생성합니다. (k값을 5로 조정하여 더 많은 문맥 참조)
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
        2. 프롬프트 (프롬프트) - 대화 맥락화
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
        2. 프롬프트 (프롬프트) - 질문 답변
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
    def get_intent_detection_prompt(): 
        """
        사용자 질문의 의도를 감지하기 위한 프롬프트입니다. LLM이 'DOCUMENTS' 또는 'GENERAL' 텍스트를 직접 반환하도록 유도합니다.
        """
        return ChatPromptTemplate.from_messages([
            ("system", "사용자의 질문 의도를 분류하세요. 문서 관련 질문이면 'DOCUMENTS', 일반적인 지식 질문이면 'GENERAL'. 답변은 오직 'DOCUMENTS' 또는 'GENERAL' 중 하나로만 하세요."),
            ("human", "{question}"), 
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
        # output_messages_key="answer"를 설정하면 딕셔너리 형태로 반환될 것으로 기대
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
    nltk.data.path.append(nltk_data_path)  # .path 속성에 append

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
    data_path = "./data"  # 문서 폴더 경로
    vectorstore = None  # 초기 벡터 저장소는 None으로 설정
    
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small') 
    
    general_llm_manager = LLMManager(model_name)
    general_llm = general_llm_manager.get_llm()

    processed_any_document = False
    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)
        
        if os.path.isfile(filepath): 
            try:
                if filename.lower().endswith(".pdf"):
                    loader = UnstructuredFileLoader(filepath) 
                elif filename.lower().endswith(".docx"):
                    loader = Docx2txtLoader(filepath)
                elif filename.lower().endswith(".pptx"):
                    loader = UnstructuredPowerPointLoader(filepath) 
                elif filename.lower().endswith(".txt"):
                    loader = TextLoader(filepath, encoding="utf-8") 
                # CSVLoader는 UnstructuredFileLoader가 처리할 수 있으므로 제거합니다.
                # elif filename.lower().endswith(".csv"): 
                #     loader = CSVLoader(filepath)
                else:
                    continue 

                single_document_list = loader.load() 
                
                if not single_document_list:
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
    
    rag_retriever = RAGRetriever(vectorstore)
    retriever = rag_retriever.get_retriever()
    llm_manager = LLMManager(model_name)
    llm = llm_manager.get_llm()
    rag_chain = RAGChain(retriever, llm)  # RAG 체인
    
    return rag_chain, general_llm 


def format_output(response):
    """
    체인 응답을 통일된 딕셔너리 형태로 포맷팅합니다.
    response는 dict 또는 AIMessage 객체일 수 있습니다.
    """
    answer = '답변을 생성할 수 없습니다.'
    context = []
    source_count = 0

    if isinstance(response, dict):
        answer = response.get('answer', answer)
        context = response.get('context', context)
        source_count = len(context)
    elif isinstance(response, AIMessage): 
        answer = response.content 
        context = [] 
        source_count = 0
    # 다른 예상치 못한 타입일 경우 문자열로 변환 (최후의 수단)
    else: 
        answer = str(response)

    return {
        'answer': answer,
        'context': context,
        'source_count': source_count
    }

# ====================================
# Streamlit UI
# ====================================

def main():
    # Session State 초기화
    if 'force_rag_keywords' not in st.session_state:
        st.session_state.force_rag_keywords = ""
    # 사용자 정의 질문 유형 설정 상태 초기화
    if 'question_type_override' not in st.session_state:
        st.session_state.question_type_override = "자동 분류"  # 기본값

    # NLTK 데이터 다운로드
    nltk_download_status = download_nltk_data()
    if nltk_download_status is None: 
        return

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
        
        # RAG 강제 활성화 키워드 입력 필드
        st.session_state.force_rag_keywords = st.text_input(
            "RAG 강제 활성화 키워드 (쉼표로 구분)",
            value=st.session_state.force_rag_keywords,
            help="여기에 입력된 키워드가 질문에 포함되면, LLM 의도 분류와 관계없이 RAG 모드로 강제 실행됩니다. (예: 문서, 보고서, 파일)"
        )
        st.markdown("---")  # 구분선 추가

        # 사용자 정의 질문 유형 선택 필드 추가
        st.session_state.question_type_override = st.radio(
            "질문 유형 설정",
            ("자동 분류", "문서 관련 강제", "일반 지식 강제"),
            index=["자동 분류", "문서 관련 강제", "일반 지식 강제"].index(st.session_state.question_type_override),
            help="질문 처리 방식을 수동으로 설정합니다. '자동 분류'는 LLM 또는 키워드에 따라 결정됩니다."
        )
        st.markdown("---")  # 구분선 추가

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

    rag_chain_wrapper, llm_for_general_qa = initialize_rag_system(model_option) 
    
    if rag_chain_wrapper is None: 
        return

    chat_history = StreamlitChatMessageHistory(key="chat_messages")
    conversational_rag_chain = rag_chain_wrapper.get_conversational_chain(chat_history)
    
    prompt_manager = PromptManager() 
    general_llm_chain_template = prompt_manager.get_general_qa_prompt() 
    
    general_qa_chain_raw = general_llm_chain_template | llm_for_general_qa  # LCEL 사용
    general_conversational_chain = RunnableWithMessageHistory(
        general_qa_chain_raw, 
        lambda session_id: chat_history, 
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer", 
    )
    
    # --- 의도 감지 체인 생성 (텍스트 반환 및 파싱으로 변경) ---
    intent_detection_prompt = prompt_manager.get_intent_detection_prompt()
    intent_detection_llm = ChatOpenAI(model=model_option, temperature=0)  # temperature=0으로 일관된 분류 유도
    
    # 프롬프트와 LLM을 직접 연결 (structured_output 사용 제거)
    intent_detection_chain_pre_invoke = intent_detection_prompt | intent_detection_llm
    # --------------------------------------------------------
    
    if not chat_history.messages:
        chat_history.add_ai_message("안녕하세요! `data` 폴더의 문서에 대해 무엇이든 물어보세요! 📚")

    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input("문서에 대해 궁금한 점을 입력해주세요..."):
        st.chat_message("human").write(prompt)

        with st.chat_message("ai"):
            with st.spinner("🧐 질문을 분석하고 답변을 생성 중입니다..."): 
                try:
                    # ★★★ 질문 유형 결정 로직 수정 ★★★
                    determined_intent = ""
                    
                    # 1. 사용자 수동 설정 우선 적용
                    if st.session_state.question_type_override == "문서 관련 강제":
                        determined_intent = "DOCUMENTS"
                        st.info("⚙️ 사용자 설정에 따라 RAG 모드로 강제 전환합니다.")
                    elif st.session_state.question_type_override == "일반 지식 강제":
                        determined_intent = "GENERAL"
                        st.info("⚙️ 사용자 설정에 따라 일반 LLM 모드로 강제 전환합니다.")
                    else:  # "자동 분류"일 경우
                        # 2. 키워드 기반 RAG 강제 활성화 로직
                        force_rag_by_keyword = False
                        if st.session_state.force_rag_keywords:
                            keywords = [k.strip().lower() for k in st.session_state.force_rag_keywords.split(',') if k.strip()]
                            for keyword in keywords:
                                if keyword in prompt.lower():
                                    force_rag_by_keyword = True
                                    break
                        
                        if force_rag_by_keyword:
                            determined_intent = "DOCUMENTS"  # 키워드가 발견되면 RAG 강제
                            st.info(f"🔑 키워드 '{', '.join(keywords)}' 감지! RAG 모드로 강제 전환합니다.")
                        else:
                            # 3. LLM 기반 의도 감지
                            try:
                                intent_response_message = intent_detection_chain_pre_invoke.invoke(
                                    {"question": prompt} 
                                )
                                determined_intent = intent_response_message.content.strip().upper() 
                            except Exception as e:
                                st.warning(f"LLM 의도 감지 중 오류 발생: {e}. 기본적으로 RAG 모드로 진행합니다.")
                                determined_intent = "DOCUMENTS"  # LLM 의도 감지 실패 시 RAG 폴백

                    final_answer = ""
                    final_context = []
                    final_source_count = 0
                    used_rag_successfully = False 

                    if determined_intent == "GENERAL": 
                        st.info("💡 일반적인 질문으로 판단하여 LLM의 일반 지식으로 답변합니다.")
                        config = {"configurable": {"session_id": "general_chat"}}
                        response_from_llm_formatted = format_output(general_conversational_chain.invoke({"input": prompt}, config))
                        final_answer = response_from_llm_formatted['answer']

                    elif determined_intent == "DOCUMENTS":
                        st.info("🔍 문서 관련 질문으로 판단하여 문서 검색 후 답변합니다.")
                        config = {"configurable": {"session_id": "rag_chat"}}
                        response_from_rag_formatted = format_output(conversational_rag_chain.invoke({"input": prompt}, config))
                        
                        rag_answer_content = response_from_rag_formatted['answer']

                        # LLM이 "문서에 관련 정보가 없습니다."를 생성하거나 컨텍스트가 없으면 폴백
                        if "문서에 관련 정보가 없습니다." in rag_answer_content or not response_from_rag_formatted.get('context'):
                            st.warning("⚠️ 문서에서 관련 정보를 찾지 못하여 LLM의 일반 지식으로 전환합니다.")
                            config = {"configurable": {"session_id": "general_chat"}} 
                            response_from_llm_formatted = format_output(general_conversational_chain.invoke({"input": prompt}, config))
                            final_answer = response_from_llm_formatted['answer']
                        else:
                            final_answer = response_from_rag_formatted['answer']
                            final_context = response_from_rag_formatted['context']
                            final_source_count = response_from_rag_formatted['source_count']
                            used_rag_successfully = True 

                    else:  # 의도 파악 실패 시 (LLM이 'DOCUMENTS' 또는 'GENERAL' 외의 것을 반환한 경우) 기본 RAG 모드로 진행
                        st.warning(f"의도 파악에 실패했습니다. (응답: {determined_intent}). 기본적으로 RAG 모드로 진행합니다.")
                        config = {"configurable": {"session_id": "rag_chat"}}
                        response_from_rag_formatted = format_output(conversational_rag_chain.invoke({"input": prompt}, config))
                        
                        rag_answer_content = response_from_rag_formatted['answer']
                        if "문서에 관련 정보가 없습니다." in rag_answer_content or not response_from_rag_formatted.get('context'):
                            st.warning("⚠️ 문서에서 관련 정보를 찾지 못하여 LLM의 일반 지식으로 전환합니다. (의도 파악 실패 후)")
                            config = {"configurable": {"session_id": "general_chat"}} 
                            response_from_llm_formatted = format_output(general_conversational_chain.invoke({"input": prompt}, config))
                            final_answer = response_from_llm_formatted['answer']
                        else:
                            final_answer = response_from_rag_formatted['answer']
                            final_context = response_from_rag_formatted['context']
                            final_source_count = response_from_rag_formatted['source_count']
                            used_rag_successfully = True

                    st.write(final_answer)

                    if used_rag_successfully:
                        with st.expander(f"📄 참고 문서 ({final_source_count}개)"):
                            if final_context: 
                                for i, doc in enumerate(final_context):
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
                                    if i < len(final_context) - 1:
                                        st.markdown("---")
                            else: 
                                st.info("답변에 참고한 문서를 찾을 수 없습니다.") 
                    else: 
                        st.info("답변에 참고한 문서가 없습니다. (일반 LLM 답변)") 

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")
                    st.info("다시 시도해주세요.")

if __name__ == "__main__":
    main()
