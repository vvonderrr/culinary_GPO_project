"""Консольный чат с кулинарным консультантом «Ням Шеф» (локальный LM Studio)."""

import re
import time

from openai import APIConnectionError, APIStatusError, OpenAI

# ---------------------------------------------------------------- настройки
BASE_URL = "http://localhost:1234/v1"
MAX_TOKENS = 400          # ответ по промпту 2-5 предложений, 1500 не нужно
TEMPERATURE = 0.7
TOP_P = 0.8
HISTORY_LIMIT = 16        # когда в истории больше N сообщений...
HISTORY_KEEP = 8          # ...оставляем последние M (чётное число: пары user/assistant)

POLITE_REPLY = "Извините, я не могу помочь с этим запросом. Пожалуйста, будьте вежливы."

# ---------------------------------------------------------- шлюз вежливости
# Если у вас есть модуль со шлюзом на KartaSlovSent, поправьте имя модуля.
# Ожидается: is_polite(text) -> True, если текст вежливый.
try:
    from polite_gate import is_polite
except ImportError:
    # Запасной вариант: быстрая проверка по корням слов, работает без датасета.
    _RUDE = re.compile(
        r"(хуй|хуе|хуё|пизд|ебан|ебат|бляд|блят|\bсука\b|сучк|тупой|тупая|"
        r"тупиц|дурак|\bдура\b|идиот|дебил|кретин|заткнись)",
        re.IGNORECASE,
    )

    def is_polite(text: str) -> bool:
        return _RUDE.search(text) is None


# ----------------------------------------------------------- системный промпт
SYSTEM_PROMPT = """Ты — дружелюбный кулинарный консультант в приложении «Ням Шеф». Помогаешь готовить: отвечаешь на вопросы по рецептам, подсказываешь замены ингредиентов, объясняешь термины.

## Как отвечать
- Если рецепт уже есть в диалоге, отвечай по нему и не переписывай его целиком.
- Если рецепта нет и спрашивают, что приготовить, предложи 2–3 блюда по одной фразе о каждом и спроси, какое расписать подробнее.
- Если данных не хватает, задай 1–2 коротких вопроса по делу («Лазанья с мясом или овощная?», «На сколько человек?»). Анкетных вопросов про опыт не задавай.
- Помогай с количеством и заменой ингредиентов, временем и признаками готовности, адаптацией (постно, менее жирно, другое число порций).
- Любой кулинарный термин сразу объясняй простыми словами.

## Стиль
- Дружелюбно и терпеливо, как друг, который учит готовить.
- Коротко: 2–5 предложений. Список — только если без него никак, не больше 5 пунктов.
- Количества указывай в граммах, миллилитрах, ложках и штуках.
- Пиши только по-русски, без английских слов; названия блюд кириллицей («паста», «карбонара»).
- Обычный текст без разметки: без символов #, ** и таблиц.
- Не больше 1–2 эмодзи за ответ.

## Безопасность
- Несъедобное или опасное (клей, мыло, стекло, гвозди): не обвиняй пользователя, по-доброму объясни, что это опасно для здоровья, и предложи готовить из настоящих продуктов.
- Сырое мясо, сырые яйца, хранение, подозрительные продукты: советуй только безопасный вариант (курицу доводить до полной готовности, продукты с плесенью выбрасывать).
- Аллергии и диеты по медицинским причинам: дай общий совет и порекомендуй уточнить у врача.
- Вопросы не про кулинарию: вежливо скажи, что помогаешь только с готовкой, и предложи вернуться к рецепту.

## Тон пользователя
- Обычное сообщение: отвечай как обычно.
- Требовательный тон без оскорблений («быстро ответь», «давай живее»): сначала полностью ответь по существу, затем в конце добавь одну короткую фразу: «И давайте общаться чуть вежливее — так готовить приятнее 😊». Не упрекай и не повторяй это дважды.
- Мат, оскорбления, угрозы: весь ответ — ровно одна фраза, без добавлений:
Извините, я не могу помочь с этим запросом. Пожалуйста, будьте вежливы.

## Примеры
Пользователь: А сколько соли класть в жаркое?
Ты: Для жаркого на 500 г курицы и 6 картофелин хватит 1–1,5 чайной ложки соли. Половину положите в середине готовки, а в конце попробуйте и досолите. Рассказать, какие специи сюда подойдут?

Пользователь: Помоги с лазаньей
Ты: С удовольствием! Лазанья будет с мясом или овощная? И листы теста у вас обычные или без варки?

Пользователь: Быстро ответь, сколько варить яйцо всмятку
Ты: 3–4 минуты после закипания воды: белок схватится, а желток останется жидким. И давайте общаться чуть вежливее — так готовить приятнее 😊"""


# ------------------------------------------------------------------ функции
def get_model_id(client: OpenAI) -> str:
    """Берёт id загруженной в LM Studio модели, чтобы не гадать с именем."""
    models = client.models.list().data
    if not models:
        raise RuntimeError("В LM Studio не загружена ни одна модель.")
    return models[0].id


def trim_history(history: list) -> None:
    """Режет историю пачкой, а не по одному сообщению.

    Почему так: LM Studio кэширует начало промпта (KV-кэш). Если убирать по
    одному сообщению каждый ход, начало меняется постоянно и кэш не работает,
    а на слабом GPU это лишние секунды. Пачкой кэш ломается редко.
    """
    if len(history) - 1 > HISTORY_LIMIT:
        del history[1:len(history) - HISTORY_KEEP]


def stream_answer(client: OpenAI, model: str, messages: list):
    """Печатает ответ потоком. Возвращает (текст, прерван_ли, статистика)."""
    start = time.perf_counter()
    first_token_at = None
    parts = []
    usage = None
    finish_reason = None
    interrupted = False

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
    )

    try:
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            # Финальный чанк с usage приходит с пустым choices: без проверки IndexError
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            piece = getattr(choice.delta, "content", None)
            if piece:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                parts.append(piece)
                print(piece, end="", flush=True)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        stream.close()

    end = time.perf_counter()
    tokens = usage.completion_tokens if usage else len(parts)  # без usage — оценка по чанкам
    gen_time = (end - first_token_at) if first_token_at else 0.0
    stats = {
        "ttft": (first_token_at - start) if first_token_at else None,
        "total": end - start,
        "tokens": tokens,
        "tps": tokens / gen_time if gen_time > 0 else 0.0,
        "truncated": finish_reason == "length",
    }
    return "".join(parts), interrupted, stats


def print_stats(stats: dict) -> None:
    ttft = f"{stats['ttft']:.1f} с" if stats["ttft"] is not None else "—"
    print(
        f"[до 1-го токена: {ttft} | всего: {stats['total']:.1f} с | "
        f"токенов: {stats['tokens']} | {stats['tps']:.1f} ток/с]"
    )
    if stats["truncated"]:
        print("[ответ обрезан по max_tokens]")


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key="not-needed")

    try:
        model = get_model_id(client)
    except APIConnectionError:
        print("❌ Нет связи с LM Studio. Запустите сервер: вкладка Developer → Start Server.")
        return
    except Exception as e:
        print(f"❌ Не удалось получить модель: {e}")
        return

    print("=" * 50)
    print(f"🤖 Консольный чат «Ням Шеф» запущен. Модель: {model}")
    print("Для выхода напишите 'выход'. Ctrl+C во время ответа останавливает генерацию.")
    print("=" * 50)

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания! Приятного аппетита!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("выход", "exit", "quit"):
            print("До свидания! Приятного аппетита!")
            break

        # Шлюз вежливости: мгновенно, без обращения к GPU и без токенов.
        # В историю такие реплики не пишем, чтобы они не засоряли контекст.
        if not is_polite(user_input):
            print(f"Ответ: {POLITE_REPLY}")
            continue

        history.append({"role": "user", "content": user_input})

        try:
            print("Ответ: ", end="", flush=True)
            answer, interrupted, stats = stream_answer(client, model, history)
            print()
            print_stats(stats)
        except APIConnectionError:
            print("\n❌ Связь с LM Studio потеряна. Проверьте, что сервер запущен.")
            history.pop()  # убираем вопрос без ответа, чат продолжается
            continue
        except APIStatusError as e:
            print(f"\n❌ Сервер вернул ошибку {e.status_code}: {e.message}")
            history.pop()
            continue
        except Exception as e:
            print(f"\n❌ Непредвиденная ошибка: {e}")
            history.pop()
            continue

        if answer.strip():
            history.append({"role": "assistant", "content": answer})
        else:
            history.pop()  # пустой ответ: не оставляем висящий вопрос
            print("⚠ Модель вернула пустой ответ, попробуйте переформулировать.")

        if interrupted:
            print("[генерация остановлена]")

        trim_history(history)


if __name__ == "__main__":
    main()