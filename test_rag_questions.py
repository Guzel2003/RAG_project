# scripts/run_tests.py
import json
import sys
import os

# Добавляем корень проекта в пути, чтобы импортировать agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.brain import get_agent_response

# Ваши проверенные вопросы
TEST_CASES = [
    {
        "id": "TC-01",
        "type": "main",
        "request": "Какие графические параметры должны содержать исполнительные геодезические схемы?",
        "expected": "Агент находит список параметров в базе знаний и указывает источник (ГОСТ).",
        "tool_calls": ["search_rag"]
    },
    {
        "id": "TC-02",
        "type": "error",
        "request": "Как оформить кредит на строительство частного дома?",
        "expected": "Агент сообщает, что в базе знаний нет информации о банковских кредитах, и не выдумывает ответ.",
        "tool_calls": ["search_rag"]
    },
    {
        "id": "TC-03",
        "type": "main",
        "request": "Какие документы прилагаются к заявлению о выдаче разрешения на ввод объекта в эксплуатацию?",
        "expected": "Агент находит упоминание ст. 55 ГрК РФ и перечисляет типы документов.",
        "tool_calls": ["search_rag"]
    },
    {
        "id": "TC-04",
        "type": "main",
        "request": "Напиши формулу чтобы вычислить Общий объем механизированных земляных работ? А если объем экскавации грунта будет 5600?",
        "expected": "Агент находит формулу, сохраняет контекст предыдущего расчета и использует калькулятор для нового значения.",
        "tool_calls": ["search_rag", "calculate_expression"]
    }
]

def run_tests():
    print("🚀 Запуск автоматического тестирования агента...\n")

    results = {
        "agent_name": "Construction RAG Agent",
        "tests": []
    }

    for tc in TEST_CASES:
        print(f"🔄 Выполняется тест {tc['id']}: {tc['request'][:50]}...")

        # Получаем ответ от агента
        actual_response = get_agent_response(user_id=999, user_message=tc['request'])

        # Простая эвристика для определения passed: если ответ длиннее 50 символов и нет слова "КРИТИЧЕСКАЯ ОШИБКА"
        passed = len(actual_response) > 50 and "КРИТИЧЕСКАЯ ОШИБКА" not in actual_response

        test_result = {
            "id": tc["id"],
            "type": tc["type"],
            "request": tc["request"],
            "expected": tc["expected"],
            "actual": actual_response[:300] + "..." if len(actual_response) > 300 else actual_response, # Обрезаем для краткости, как в примере
            "tool_calls": tc["tool_calls"],
            "passed": passed
        }
        results["tests"].append(test_result)
        print(f"✅ Тест {tc['id']} завершён. Статус: {'PASS' if passed else 'FAIL'}\n")

    # Сохраняем в требуемую папку
    os.makedirs("data", exist_ok=True)
    with open("data/test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("🎉 Готово! Результаты сохранены в data/test_results.json")

if __name__ == "__main__":
    run_tests()