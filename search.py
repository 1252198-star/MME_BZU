

import os
import re
import chromadb

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. INITIALIZATION
# ==========================================

print("Loading AI model...")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_collection(name="bzu_documents")

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

groq_client = Groq(api_key=api_key)

GROQ_MODEL = "openai/gpt-oss-120b"

print("UniGuideAI is ready!")


# ==========================================
# 2. CONVERSATION MEMORY
# ==========================================

conversation_history = []


def get_history_text():

    history_text = ""

    for turn in conversation_history[-4:]:

        if not isinstance(turn, dict):
            continue

        student_question = (
            turn.get("question")
            or turn.get("content")
            or turn.get("message")
            or ""
        )

        previous_answer = turn.get("answer", "")

        if turn.get("role") == "user":
            student_question = turn.get("content", "")
            previous_answer = ""

        elif turn.get("role") in ("bot", "assistant"):
            previous_answer = (
                turn.get("content")
                or turn.get("answer")
                or ""
            )

        history_text += f"""
Student: {student_question}
UniGuideAI: {previous_answer}
"""

    return history_text


# ==========================================
# 3. TEXT NORMALIZATION
# ==========================================

def normalize(text):

    text = str(text).lower()

    replacements = [
        ("أ", "ا"),
        ("إ", "ا"),
        ("آ", "ا"),
        ("ى", "ي"),
        ("ة", "ه"),
        ("ؤ", "و"),
        ("ئ", "ي")
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    return text


# ==========================================
# 4. FOLLOW-UP QUESTION REFORMULATION
# ==========================================

def rewrite_followup_question(question):

    if not question:
        return question

    question = question.strip()

    print(f"Search question: {question}")

    return question


# ==========================================
# 5. QUESTION INTENT
# ==========================================

def is_withdrawal_question(question):

    q = normalize(question)

    terms = [
        "سحب",
        "الانسحاب",
        "انسحاب",
        "انسحب",
        "منسحب",
        "withdrawal",
        "drop course"
    ]

    return any(term in q for term in terms)


def is_graduation_question(question):

    q = normalize(question)

    terms = [
        "تخرج",
        "التخرج",
        "graduation",
        "graduate"
    ]

    return any(term in q for term in terms)


def is_cs_question(question):

    q = normalize(question)

    return (
        "علم الحاسوب" in q
        or "علوم الحاسوب" in q
        or "computer science" in q
    )


# ==========================================
# 6. SEMANTIC SEARCH
# ==========================================

def semantic_search(question, limit=20):

    embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=limit
    )

    found = []

    if not results["documents"] or not results["documents"][0]:
        return found

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        found.append({
            "text": document,
            "page": metadata.get("page", 0),
            "source": metadata.get("source", "BZU.pdf"),
            "score": max(0, 10 - distance * 10),
            "distance": distance,
            "method": "Semantic"
        })

    return found


# ==========================================
# 7. KEYWORD SEARCH
# ==========================================

def keyword_search(question):

    data = collection.get(
        include=["documents", "metadatas"]
    )

    q = normalize(question)

    course_codes = re.findall(
        r"\b[A-Za-z]{2,}\d{2,}\b",
        question
    )

    withdrawal = is_withdrawal_question(question)
    graduation = is_graduation_question(question)
    cs_question = is_cs_question(question)

    words = [
        word for word in q.split()
        if len(word) >= 3
    ]

    withdrawal_pages = {
        19: 1000,
        20: 950,
        18: 500,
        16: 350
    }

    found = []

    for i, document in enumerate(data["documents"]):

        metadata = data["metadatas"][i]
        page = metadata.get("page", 0)

        doc = normalize(document)
        score = 0

        if course_codes:

            code_hits = sum(
                1 for code in course_codes
                if normalize(code) in doc
            )

            if code_hits == 0:
                continue

            score += code_hits * 150

            for code in course_codes:

                position = doc.find(normalize(code))

                if position != -1:

                    context = doc[position:position + 400]

                    if "المتطلب السابق" in context:
                        score += 100

        else:

            for word in words:

                if word in doc:
                    score += 2

            if withdrawal:

                if page in withdrawal_pages:
                    score += withdrawal_pages[page]

                important_phrases = [
                    ("الانسحاب من المساقات", 300),
                    ("الانسحاب من مساقات", 250),
                    ("اخر يوم تدريس", 100),
                    ("تعديل التسجيل", 80),
                    ("ساعات التخصص", 100),
                    ("غير متفرغ", 80),
                    ("موافقة من عميد", 60),
                    ("علامة w", 60),
                    ("الالتزامات الماليه", 40)
                ]

                for phrase, points in important_phrases:

                    if normalize(phrase) in doc:
                        score += points

            if graduation:

                if "شروط التخرج التالية" in doc:
                    score += 150

                if "الحد الادنى للساعات" in doc:
                    score += 80

                if "معدل تراكمي" in doc:
                    score += 80

            if cs_question:

                if "علم الحاسوب" in doc:
                    score += 100

                if "بكالوريوس" in doc:
                    score += 50

                if "متطلبات التخصص" in doc:
                    score += 40

                if "شروط القبول" in doc:
                    score += 100

                if "القبول في التخصص" in doc:
                    score += 100

                if "معدلات القبول" in doc:
                    score += 100

        if score > 0:

            found.append({
                "text": document,
                "page": page,
                "source": metadata.get("source", "BZU.pdf"),
                "score": score,
                "distance": None,
                "method": "Keyword"
            })

    found.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return found


# ==========================================
# 8. COURSE PREREQUISITE SEARCH
# ==========================================

def is_prerequisite_question(question):

    q = normalize(question)

    terms = [
        "المتطلب السابق",
        "المتطلبات السابقه",
        "متطلب سابق",
        "متطلبات سابقه",
        "شو متطلب",
        "ما متطلب",
        "prerequisite",
        "prerequisites"
    ]

    return any(term in q for term in terms)


def course_prerequisite_search(course_code):

    data = collection.get(
        include=["documents", "metadatas"]
    )

    course_code = course_code.upper()

    found = []

    course_pattern = re.compile(
        r"^\s*" + re.escape(course_code) + r"\b",
        re.IGNORECASE
    )

    any_course_pattern = re.compile(
        r"^\s*[A-Za-z]{2,}\d{2,}\b",
        re.IGNORECASE
    )

    for document, metadata in zip(
        data["documents"],
        data["metadatas"]
    ):

        lines = document.splitlines()

        for i, line in enumerate(lines):

            line = line.strip()

            # نتأكد أن رقم المساق في بداية السطر
            if not course_pattern.search(line):
                continue

            relevant_lines = [line]

            # نجمع وصف المساق حتى المساق التالي
            for next_line in lines[i + 1:]:

                next_line = next_line.strip()

                if not next_line:
                    continue

                # نتوقف عند بداية مساق جديد
                if any_course_pattern.match(next_line):
                    break

                relevant_lines.append(next_line)

                # إذا وجدنا المتطلب السابق نتوقف
                if "المتطلب السابق" in normalize(next_line):
                    break

            snippet = "\n".join(relevant_lines)

            # لا نقبل النتيجة إلا إذا وجدنا
            # عبارة المتطلب السابق ضمن وصف المساق نفسه
            if "المتطلب السابق" not in normalize(snippet):
                continue

            found.append({
                "text": snippet,
                "page": metadata.get("page", 0),
                "source": metadata.get("source", "BZU.pdf"),
                "score": 1000,
                "distance": None,
                "method": "Exact Course Prerequisite"
            })

    # إزالة النتائج المكررة
    unique = {}

    for result in found:

        key = (
            result["page"],
            result["text"]
        )

        unique[key] = result

    results = list(unique.values())

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


# ==========================================
# 9. HYBRID SEARCH
# ==========================================

def hybrid_search(question):

    course_codes = re.findall(
        r"\b[A-Za-z]{2,}\d{2,}\b",
        question
    )

    # بحث دقيق عن المتطلب السابق لمساق محدد
    if course_codes and is_prerequisite_question(question):

        course_code = course_codes[0]

        exact_results = course_prerequisite_search(
            course_code
        )

        if exact_results:

            print(
                f"Exact prerequisite search: {course_code}"
            )

            return exact_results[:5]

    keyword = keyword_search(question)

    if course_codes:
        return keyword[:12]

    semantic = semantic_search(question, limit=20)

    combined = {}

    for result in semantic + keyword:

        key = (
            result["page"],
            result["text"]
        )

        if key not in combined:

            combined[key] = result.copy()

        else:

            existing = combined[key]

            existing["score"] += result["score"]

            if result["method"] == "Keyword":
                existing["method"] = "Hybrid"

            if existing["distance"] is None:
                existing["distance"] = result["distance"]

    results = list(combined.values())

    if is_withdrawal_question(question):

        priority_pages = {
            19: 10000,
            20: 9000,
            18: 5000,
            16: 3000
        }

        for result in results:

            page = result["page"]

            if page in priority_pages:
                result["score"] += priority_pages[page]

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:12]


# ==========================================
# 10. LOCAL FALLBACK ANSWER
# ==========================================

def local_answer(question, results):

    if not results:
        return "ما بعرف، المعلومة غير متوفرة في الوثائق المتاحة."

    q = normalize(question)

    words = [
        word for word in q.split()
        if len(word) >= 3
    ]

    selected_sentences = []
    seen = set()

    for result in results[:12]:

        text = result.get("text", "")

        sentences = re.split(
            r'(?<=[.!؟])\s+|\n+',
            text
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) < 20:
                continue

            normalized_sentence = normalize(sentence)

            score = sum(
                1 for word in words
                if word in normalized_sentence
            )

            if score == 0:
                continue

            key = normalized_sentence

            if key in seen:
                continue

            seen.add(key)

            selected_sentences.append({
                "text": sentence,
                "score": score,
                "page": result.get("page"),
                "source": result.get("source", "BZU.pdf")
            })

    selected_sentences.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    if not selected_sentences:
        return (
            "وجدت معلومات في وثائق الجامعة، "
            "لكن لم أتمكن من تحديد نص كافٍ للإجابة بدقة. "
            "يرجى إعادة صياغة السؤال."
        )

    answer = "حسب وثائق جامعة بيرزيت:\n\n"

    for item in selected_sentences[:8]:

        answer += f"• {item['text']}\n\n"

    answer += "\nالمصادر:\n"

    added_sources = set()

    for item in selected_sentences[:8]:

        source = (
            item["source"],
            item["page"]
        )

        if source not in added_sources:

            added_sources.add(source)

            answer += (
                f"• {source[0]} - الصفحة {source[1]}\n"
            )

    answer += (
        "\nملاحظة: هذه إجابة مستخرجة مباشرة "
        "من نصوص وثائق الجامعة."
    )

    return answer


# ==========================================
# 11. AI ANSWER GENERATION - GROQ
# ==========================================

def generate_answer(question, results):

    if not results:
        return "ما بعرف، المعلومة غير متوفرة في الوثائق المتاحة."

    context = ""

    for i, result in enumerate(results[:12], start=1):

        context += f"""
SOURCE {i}
File: {result['source']}
Page: {result['page']}
Content:
{result['text']}
-------------------------
"""

    history_text = get_history_text()

    prompt = f"""
You are UniGuideAI, an academic assistant
specialized in Birzeit University.

Answer the student's question using ONLY the
university document excerpts below.

Previous conversation:
{history_text}

Latest question:
{question}

STRICT RULES:

1. Use only the provided document excerpts as evidence.
2. Never invent university regulations or requirements.
3. Never use outside knowledge.
4. Previous AI answers are context only, not evidence.
5. Distinguish admission requirements from graduation requirements.
6. Answer in the student's language.
7. Use clear and simple language.
8. Mention the source file and page number.
9. If sources conflict, explain the conflict.
10. If the documents do not contain enough information,
    say that the information is unavailable.
11. If the question asks for a course prerequisite,
    identify the prerequisite of that exact course.
12. Do not confuse a course with courses that list it
    as their own prerequisite.
13. For prerequisite questions, use the course description
    and the explicit "المتطلب السابق" statement.
14. Do not infer a prerequisite from a different course's row.

UNIVERSITY DOCUMENT EXCERPTS:
{context}

ANSWER:
"""

    try:

        print("Connecting to Groq...")

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1200
        )

        answer = response.choices[0].message.content

        if not answer:

            print("Groq returned an empty answer.")

            return local_answer(question, results)

        return answer.strip()

    except Exception as e:

        print("Groq API Error:", str(e))

        print("Generating local fallback answer...")

        return local_answer(question, results)


# ==========================================
# 12. MAIN APPLICATION
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("Welcome to UniGuideAI")
    print("Birzeit University Academic Assistant")
    print("=" * 60)

    while True:

        question = input(
            "\nAsk your academic question (exit to stop): "
        )

        if question.strip().lower() == "exit":

            print("\nGoodbye!")
            break

        if not question.strip():

            print("Please enter a question.")
            continue

        try:

            print("\nUnderstanding your question...")

            search_question = rewrite_followup_question(question)

            print("\nSearching university documents...")

            results = hybrid_search(search_question)

            print("\nRetrieved pages:")

            for result in results:

                print(
                    f"Page {result['page']} | "
                    f"Score: {result['score']} | "
                    f"{result['method']}"
                )

            print("\nGenerating answer...")

            answer = generate_answer(
                question,
                results
            )

            print("\n" + "=" * 60)
            print("UniGuideAI Answer")
            print("=" * 60)

            print(answer)

            print("=" * 60)

            conversation_history.append({
                "question": question,
                "answer": answer
            })

            conversation_history[:] = conversation_history[-10:]

        except Exception as e:

            print("\nAn error occurred:")
            print(e)
