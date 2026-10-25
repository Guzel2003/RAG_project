import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.brain import get_agent_response

TEST_CASES = [
    {
        "id": "TC-01",
        "type": "main",
        "request": "Какие графические параметры должны содержать исполнительные геодезические схемы?",
        "expected": "Агент находит список параметров и вызывает search_rag",
        "required_tools": ["search_rag"]
    },
    {
        "id": "TC-02",
        "type": "error",
        "request": "Как оформить кредит на строительство частного дома?",
        "expected": "Агент сообщает об отсутствии информации, не галлюцинирует",
        "required_tools": ["search_rag"]
    },
    {
        "id": "TC-03",
        "type": "main_calc",
        "request": "Посчитай 5600 + 7903",
        "expected": "Агент использует калькулятор и выдает правильный результат 13503",
        "required_tools": ["calculate_expression"]
    }
]


def run_tests():
    print("🚀 Запуск честного тестирования...\n")
    results = {"agent_name": "Construction RAG Agent", "tests": []}

    for tc in TEST_CASES:
        print(f"🔄 Тест {tc['id']}: {tc['request'][:40]}...")

        # Теперь мы получаем словарь с ответом и реальными инструментами!
        response_data = get_agent_response(user_id=999, user_message=tc['request'])
        actual_answer = response_data["answer"]
        actual_tools = response_data["tools_called"]

        # Проверяем, что ответ не является ошибкой
        is_valid_length = len(actual_answer) > 20 and "КРИТИЧЕСКАЯ ОШИБКА" not in actual_answer

        # Проверяем, что вызваны нужные инструменты
        tools_match = all(tool in actual_tools for tool in tc["required_tools"])

        passed = is_valid_length and tools_match

        test_result = {
            "id": tc["id"],
            "type": tc["type"],
            "request": tc["request"],
            "expected": tc["expected"],
            "actual": actual_answer[:200] + "..." if len(actual_answer) > 200 else actual_answer,
            "tool_calls": actual_tools,  # <-- ТЕПЕРЬ ЗДЕСЬ РЕАЛЬНЫЕ ДАННЫЕ!
            "passed": passed
        }
        results["tests"].append(test_result)
        print(f"✅ Статус: {'PASS' if passed else 'FAIL'}. Инструменты: {actual_tools}\n")

    os.makedirs("data", exist_ok=True)
    with open("data/test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("🎉 Готово! Результаты сохранены в data/test_results.json")


if __name__ == "__main__":
    run_tests()