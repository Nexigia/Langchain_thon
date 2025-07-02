# 🧭 메모리네비 (MemoryNavi)
> **치매 관련 정보 제공 AI 챗봇**
> 메모리네비는 고령자를 위한 치매 제도 및 복지 정보 탐색을 돕는 문서 기반 AI 챗봇입니다.      
> PDF 문서를 벡터화하여 최신 내용을 자동 분석하고, RAG 기술 기반으로 질문 의도에 맞는 신뢰도 높은 맞춤형 답변을 제공합니다.    
> CSS를 사용하여 직관적이고 큰 글씨 UI로 어르신이 쉽게 이용할 수 있도록 설계되었습니다.    
> Streamlit 기반 웹 인터페이스로 배포하였습니다.

---

## 주요 기능
- PDF 문서 기반 RAG(Retrieval-Augmented Generation) 시스템
- 치매 관련 국가 제도, 복지, 의료 정보 자동 응답
- OpenAI GPT-4o-mini 기반 대화형 AI
- 사용자 입력/AI 응답 UI 커스터마이징 (고령자 친화형 폰트 및 구성)
- Streamlit 기반 웹 인터페이스
- 대화 히스토리 기억 및 반영

---

## 시스템 아키텍처

<table style="border-collapse: collapse; width: 100%; font-size: 16px;">
  <thead style="background-color: #f2f2f2;">
    <tr>
      <th style="text-align: left; padding: 8px;">구성 요소</th>
      <th style="text-align: left; padding: 8px;">설명</th>
    </tr>
  </thead>
  <tbody style="background-color: #ffffff;">
    <tr>
      <td style="padding: 8px;"><strong>UI</strong></td>
      <td style="padding: 8px;">Streamlit 기반 채팅 인터페이스</td>
    </tr>
    <tr>
      <td style="padding: 8px;"><strong>문서 로더</strong></td>
      <td style="padding: 8px;"><code>PyPDFLoader</code> (PDF → 텍스트 분할)</td>
    </tr>
    <tr>
      <td style="padding: 8px;"><strong>텍스트 분할기</strong></td>
      <td style="padding: 8px;"><code>RecursiveCharacterTextSplitter</code> (chunk_size=500, overlap=100)</td>
    </tr>
    <tr>
      <td style="padding: 8px;"><strong>임베딩 모델</strong></td>
      <td style="padding: 8px;">OpenAI <code>text-embedding-3-small</code></td>
    </tr>
    <tr>
      <td style="padding: 8px;"><strong>벡터 DB</strong></td>
      <td style="padding: 8px;">FAISS (로컬 저장 및 로딩 지원)</td>
    </tr>
    <tr>
      <td style="padding: 8px;"><strong>LLM</strong></td>
      <td style="padding: 8px;">OpenAI <code>gpt-4o-mini</code></td>
    </tr>
    <tr>
      <td style="padding: 8px;"><strong>Retriever</strong></td>
      <td style="padding: 8px;">문서 검색 + 대화 이력 기반 <code>create_history_aware_retriever()</code></td>
    </tr>
    <tr>
      <td style="padding: 8px;"><strong>QA Prompt</strong></td>
      <td style="padding: 8px;">문서 기반 응답 생성 프롬프트 (<code>ChatPromptTemplate</code>)</td>
    </tr>
    <tr>
      <td style="padding: 8px;"><strong>Chain 구성</strong></td>
      <td style="padding: 8px;"><code>create_retrieval_chain()</code> + <code>RunnableWithMessageHistory</code></td>
    </tr>
  </tbody>
</table>
