import requests
import sys
import pytest
import os
import time
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from tema_3_evaluation.groq_llm import GroqDeepEval

# foloseste UTF-8 pentru stdout ca sa evite erori de codare
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"

def _post_chat_with_retry(message: str, max_retries: int = 2) -> requests.Response:
    last_response = None
    for attempt in range(max_retries + 1):
        response = requests.post(
            f"{BASE_URL}/chat/",
            json={"message": message},
            timeout=90,
        )
        last_response = response

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if response.status_code != 504 or payload.get("detail") != "Raspunsul de chat a expirat":
            return response

        if attempt < max_retries:
            time.sleep(2)

    return last_response


@pytest.fixture(scope="session", autouse=True)
def ensure_server_is_running():
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"Serverul nu ruleaza pe {BASE_URL}. Porneste `uvicorn app.main:app --reload`. ({exc})")


@pytest.fixture(scope="session")
def llm_judge_model():
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("Seteaza variabila de mediu GROQ_API_KEY pentru testele LLM-as-a-Judge.")
    return GroqDeepEval()


def test_root_endpoint():
    response = requests.get(f"{BASE_URL}/", timeout=10)
    assert response.status_code == 200
    assert response.json() == {"message": "Salut, RAG Assistant ruleaza!"}


def test_chat_relevant_scenario_llm_judge(llm_judge_model):
    message = (
        "Cum pot implementa skill packs pentru industria telecom intr-un chatbot AI "
        "de support center care foloseste customer memory?"
    )
    response = _post_chat_with_retry(message)
    assert response is not None
    assert response.status_code == 200

    payload = response.json()
    assistant_answer = str(payload.get("response", "")).strip()
    assert assistant_answer, "Endpoint-ul /chat/ nu a returnat un raspuns text valid."

    evaluator = GEval(
        name="Relevanta raspunsului pentru Support Center",
        criteria="""
        Raspunsul trebuie sa trateze explicit cerinta din domeniul Support Center AI.
        Trebuie sa mentioneze idei relevante precum skill packs, customer memory,
        workflow de suport sau arhitectura chatbot. Penalizeaza raspunsuri vagi.
        """,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=llm_judge_model,
    )

    test_case = LLMTestCase(
        input=message,
        actual_output=assistant_answer,
    )

    evaluator.measure(test_case)
    assert evaluator.score >= 0.7, (
        "Scorul de relevanta este prea mic pentru un scenariu pozitiv. "
        f"score={evaluator.score:.2f}, reason={evaluator.reason}"
    )


def test_chat_negative_off_topic_llm_judge(llm_judge_model):
    message = "Care este reteta pentru tiramisu?"
    response = _post_chat_with_retry(message)
    assert response is not None
    assert response.status_code == 200

    payload = response.json()
    assistant_answer = str(payload.get("response", "")).strip()
    assert assistant_answer, "Endpoint-ul /chat/ nu a returnat un raspuns text valid."

    evaluator = GEval(
        name="Respectarea domeniului in scenariu negativ",
        criteria="""
        Pentru o intrebare off-topic, raspunsul trebuie sa semnaleze clar ca subiectul
        nu este in domeniul chatbotului de Support Center si sa redirectioneze
        utilizatorul spre teme relevante (skill packs, customer memory, suport clienti).
        Penalizeaza raspunsurile care trateaza direct subiectul culinar.
        """,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=llm_judge_model,
    )

    test_case = LLMTestCase(
        input=message,
        actual_output=assistant_answer,
    )

    evaluator.measure(test_case)
    assert evaluator.score >= 0.7, (
        "Raspunsul nu respecta suficient domeniul in testul negativ. "
        f"score={evaluator.score:.2f}, reason={evaluator.reason}"
    )
