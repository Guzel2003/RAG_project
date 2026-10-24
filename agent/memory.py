# memory.py
# Простая сессионная память для хранения истории диалога каждого пользователя

class SessionMemory:
    def __init__(self):
        # Словарь, где ключ - это ID пользователя Telegram, а значение - список сообщений
        self.histories = {}

    def add_message(self, user_id: int, role: str, content: str):
        """Добавляет сообщение в историю конкретного пользователя."""
        if user_id not in self.histories:
            self.histories[user_id] = []

        self.histories[user_id].append({"role": role, "content": content})

        # Ограничиваем историю последними 10 сообщениями, чтобы не переполнять память
        if len(self.histories[user_id]) > 10:
            self.histories[user_id] = self.histories[user_id][-10:]

    def get_history(self, user_id: int) -> list:
        """Возвращает историю диалога для пользователя."""
        return self.histories.get(user_id, [])

    def clear_history(self, user_id: int):
        """Очищает историю для пользователя (например, по команде /start)."""
        if user_id in self.histories:
            del self.histories[user_id]


# Создаем глобальный экземпляр памяти, который будем импортировать в других файлах
memory = SessionMemory()