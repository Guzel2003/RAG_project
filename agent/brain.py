# agent/brain.py
import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
from agent.memory import memory
from agent.tools import search_rag, calculate_expression

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Обновленный промпт для работы в цикле
SYSTEM_PROMPT = """Ты — профессиональный инженер-строитель и AI-агент.
Твоя задача: рассчитывать элементы конструкций, используя ТОЛЬКО формулы из базы знаний и точные вычисления.

Доступные инструменты (tools):
1. `search_rag(query: str)` — ищет фрагменты документов в базе Qdrant. Возвращает 'context' с формулами и 'sources'.
2. `calculate_expression(expression: str)` — вычисляет математическое выражение (например, "(200 * 4**2) / 8").

АЛГОРИТМ ДЕЙСТВИЙ:
1. Если пользователь просит расчет, сначала вызови `search_rag` с кратким запросом (например, "формула изгибающего момента балки").
2. Внимательно прочитай возвращенный 'context'. Найди в нем нужную формулу и переменные.
3. Извлеки из запроса пользователя числовые значения.
4. Сформулируй финальный профессиональный ответ. Если в полученном контексте нет точной формулы или данных, честно напиши: 
   "В найденных фрагментах базы знаний упоминается этот расчет, но точная формула обрезана. Однако известно, что..." и приведи доступные данные.
5. Сформулируй финальный профессиональный ответ, указывая источник (из 'sources') и показывая ход расчета.

ПРАВИЛА ВЫЗОВА ИНСТРУМЕНТА (СТРОГО СОБЛЮДАЙ ЭТОТ ФОРМАТ):
Чтобы вызвать инструмент, ответь ТОЛЬКО так:
TOOL_CALL: <имя_инструмента>
ARGS: <строго_валидный_JSON_объект_в_фигурных_скобках>

ПРИМЕР ПРАВИЛЬНОГО ВЫЗОВА:
TOOL_CALL: calculate_expression
ARGS: {"expression": "5000 + 7656"}

НЕ ПИШИ просто строку в ARGS. Всегда используй фигурные скобки {} и имя параметра!
"""


def get_agent_response(user_id: int, user_message: str) -> str:
    # Добавляем запрос пользователя в память
    memory.add_message(user_id, "user", user_message)

    # Ограничиваем количество шагов, чтобы избежать бесконечного цикла
    max_steps = 5

    for step in range(max_steps):
        print(f"\n🔄 ШАГ {step + 1}: Отправляю запрос к LLM...")

        # Формируем контекст из системного промпта и всей истории
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + memory.get_history(user_id)

        try:
            response = client.chat.completions.create(
                model="recursive/flint-v1.0:free",
                messages=messages,
                temperature=0.1,
            )

            agent_reply = response.choices[0].message.content
            print(f"💬 Черновик LLM:\n{agent_reply[:150]}...")

            # Ищем вызов инструмента
            tool_match = re.search(r'TOOL_CALL:\s*([a-zA-Z0-9_]+)\s*ARGS:\s*(\{.*\})', agent_reply,
                                   re.IGNORECASE | re.DOTALL)

            if tool_match:
                # ЭТО ВЫЗОВ ИНСТРУМЕНТА!
                tool_name = tool_match.group(1).strip()
                args_str = tool_match.group(2).strip()
                print(f"🔧 Обнаружен вызов инструмента: {tool_name}")

                try:
                    kwargs = json.loads(args_str)

                    # Выполняем инструмент
                    if tool_name == "search_rag":
                        tool_result = search_rag(**kwargs)
                    elif tool_name == "calculate_expression":
                        tool_result = calculate_expression(**kwargs)
                    else:
                        tool_result = {"status": "error", "error": f"Неизвестный инструмент: {tool_name}"}

                    print(f"✅ Результат инструмента: {tool_result}")

                    if tool_result.get("status") == "error":
                        return f"❌ Ошибка инструмента: {tool_result.get('error')}. Проверьте данные или перефразируйте запрос."

                    # === ИЗМЕНЕНИЕ ЗДЕСЬ: Формируем четкую инструкцию вместо сырого JSON ===
                    tool_response_text = (
                        f"РЕЗУЛЬТАТ ВЫПОЛНЕНИЯ ИНСТРУМЕНТА '{tool_name}':\n"
                        f"{json.dumps(tool_result, ensure_ascii=False, indent=2)}\n\n"
                        f"ТВОЯ ЗАДАЧА: Проанализируй эти данные.\n"
                        f"1. Если в них есть точный ответ на вопрос пользователя, сформулируй его четко и профессионально.\n"
                        f"2. Если в данных нет полной информации (например, текст формулы в контексте обрезан), честно напиши: "
                        f"'В найденных фрагментах базы знаний упоминается этот расчет, но точная формула в данном фрагменте обрезана. "
                        f"Однако известно, что...' и приведи те данные, которые есть."
                    )

                    # Добавляем эту инструкцию в память как сообщение от "пользователя" (системы),
                    # чтобы заставить модель сгенерировать финальный ответ ассистента
                    memory.add_message(user_id, "user", tool_response_text)

                    # Цикл продолжится (step + 1), и LLM получит этот результат!
                    continue

                except json.JSONDecodeError:
                    return f"❌ Ошибка: не удалось распознать аргументы инструмента. Сырые данные: {args_str}"
                except Exception as e:
                    return f"❌ Ошибка выполнения инструмента: {str(e)}"

            else:
                # ЭТО ФИНАЛЬНЫЙ ОТВЕТ! (Нет TOOL_CALL, значит модель закончила думать)
                print("✅ Финальный ответ сформирован.")
                memory.add_message(user_id, "assistant", agent_reply)
                return agent_reply

        except Exception as e:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {type(e).__name__}: {e}")
            return f"Извините, техническая ошибка: {str(e)}"

    return "⚠️ Превышено максимальное количество шагов рассуждения. Попробуйте упростить запрос."