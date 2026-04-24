"""Tests for dictate_realtime logic: hallucination filter, commands, diff, clean."""
import re
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ---- import logic from dictate_realtime without running main ----

HALLUCINATIONS = [
    "спасибо за внимание", "спасибо за просмотр",
    "подписывайтесь на канал", "продолжение следует",
    "до новых встреч", "до свидания", "с вами был",
    "редактор субтитров", "корректор", "добро пожаловать",
    "music", "you", "thank you", "the end",
]


def is_hallucination(text: str) -> bool:
    t = text.lower().strip().rstrip(".")
    if len(t) < 3:
        return True
    return any(h in t for h in HALLUCINATIONS)


def _clean(text: str) -> str:
    return text.strip().rstrip(".,!?;:").strip()


COMMANDS: list[tuple[re.Pattern, str]] = []


def cmd(pattern: str, name: str):
    COMMANDS.append((re.compile(pattern, re.IGNORECASE), name))


cmd(r"^удали(ть|те)?\s+слово$", "удалить слово")
cmd(r"^удали(ть|те)?\s+(последнее\s+)?предложение$", "удалить предложение")
cmd(r"^удали(ть|те)?\s+строк[уа]$", "удалить строку")
cmd(r"^удали(ть|те)?\s+вс[её]$", "удалить всё")
cmd(r"^отмен(ить|а|и|ите)$", "отмена")
cmd(r"^нов(ая|ую)\s+строк[уа]$", "новая строка")
cmd(r"^(enter|энтер|ввод)$", "enter")
cmd(r"^нов(ый|ому)\s+(абзац|параграф)$", "новый абзац")
cmd(r"^таб(уляция)?$", "таб")


def match_command_full(text: str) -> str | None:
    clean = _clean(text)
    for pattern, name in COMMANDS:
        if pattern.match(clean):
            return name
    return None


def match_command_trailing(text: str) -> tuple[str, str | None]:
    words = text.split()
    for n in range(1, min(6, len(words) + 1)):
        tail = _clean(" ".join(words[-n:]))
        for pattern, name in COMMANDS:
            if pattern.match(tail):
                remaining = " ".join(words[:-n]).strip()
                return remaining, name
    return text, None


def compute_diff(old: str, new: str) -> tuple[int, str]:
    common = 0
    for a, b in zip(old, new):
        if a == b:
            common += 1
        else:
            break
    return len(old) - common, new[common:]


# ---- TESTS ----

passed = 0
failed = 0


def check(name: str, actual, expected):
    global passed, failed
    if actual == expected:
        passed += 1
        print(f"  [OK]   {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")
        print(f"         expected: {expected!r}")
        print(f"         actual:   {actual!r}")


# ===== HALLUCINATION FILTER =====
print("\n=== Hallucination Filter ===")

check("silence: empty", is_hallucination(""), True)
check("silence: short", is_hallucination("ок"), True)
check("silence: single char", is_hallucination("а"), True)

check("halluc: спасибо за внимание", is_hallucination("Спасибо за внимание."), True)
check("halluc: спасибо за внимание (no dot)", is_hallucination("спасибо за внимание"), True)
check("halluc: редактор субтитров", is_hallucination("Редактор субтитров А.Синецкая"), True)
check("halluc: корректор", is_hallucination("Корректор А.Кулакова"), True)
check("halluc: добро пожаловать на канал", is_hallucination("Добро пожаловать на наш канал!"), True)
check("halluc: music", is_hallucination("Music"), True)
check("halluc: продолжение следует", is_hallucination("Продолжение следует..."), True)
check("halluc: the end", is_hallucination("The End."), True)
check("halluc: до свидания", is_hallucination("До свидания!"), True)
check("halluc: thank you", is_hallucination("Thank you."), True)
check("halluc: с вами был Игорь", is_hallucination("С вами был Игорь Негода."), True)

check("legit: привет", is_hallucination("Привет"), False)
check("legit: как дела", is_hallucination("Как дела?"), False)
check("legit: удалить слово", is_hallucination("удалить слово"), False)
check("legit: normal sentence", is_hallucination("Я хочу написать письмо."), False)
check("legit: long text", is_hallucination("Сегодня хорошая погода и я иду гулять"), False)
check("legit: числа", is_hallucination("12345"), False)

# ===== CLEAN FUNCTION =====
print("\n=== Clean Function ===")

check("clean: trailing dot", _clean("Привет."), "Привет")
check("clean: trailing comma", _clean("Привет,"), "Привет")
check("clean: trailing excl", _clean("Привет!"), "Привет")
check("clean: trailing question", _clean("Привет?"), "Привет")
check("clean: trailing multi", _clean("Привет..."), "Привет")
check("clean: spaces", _clean("  Привет  "), "Привет")
check("clean: no change", _clean("Привет"), "Привет")
check("clean: empty", _clean(""), "")
check("clean: colon", _clean("Вот:"), "Вот")
check("clean: semicolon", _clean("Вот;"), "Вот")

# ===== COMMAND FULL MATCH =====
print("\n=== Command Full Match ===")

check("cmd: удалить слово", match_command_full("удалить слово"), "удалить слово")
check("cmd: удали слово", match_command_full("удали слово"), "удалить слово")
check("cmd: удалите слово", match_command_full("удалите слово"), "удалить слово")
check("cmd: Удалить слово", match_command_full("Удалить слово"), "удалить слово")
check("cmd: Удалить слово.", match_command_full("Удалить слово."), "удалить слово")
check("cmd: УДАЛИТЬ СЛОВО", match_command_full("УДАЛИТЬ СЛОВО"), "удалить слово")

check("cmd: удалить предложение", match_command_full("удалить предложение"), "удалить предложение")
check("cmd: удалить последнее предложение", match_command_full("удалить последнее предложение"), "удалить предложение")
check("cmd: удалите предложение", match_command_full("удалите предложение"), "удалить предложение")

check("cmd: удалить строку", match_command_full("удалить строку"), "удалить строку")
check("cmd: удалить строка", match_command_full("удалить строка"), "удалить строку")

check("cmd: удалить всё", match_command_full("удалить всё"), "удалить всё")
check("cmd: удалить все", match_command_full("удалить все"), "удалить всё")
check("cmd: удали все.", match_command_full("удали все."), "удалить всё")

check("cmd: отменить", match_command_full("отменить"), "отмена")
check("cmd: отмена", match_command_full("отмена"), "отмена")
check("cmd: отмени", match_command_full("отмени"), "отмена")
check("cmd: отмените", match_command_full("отмените"), "отмена")
check("cmd: Отменить.", match_command_full("Отменить."), "отмена")

check("cmd: новая строка", match_command_full("новая строка"), "новая строка")
check("cmd: новую строку", match_command_full("новую строку"), "новая строка")

check("cmd: enter", match_command_full("enter"), "enter")
check("cmd: энтер", match_command_full("энтер"), "enter")
check("cmd: ввод", match_command_full("ввод"), "enter")

check("cmd: новый абзац", match_command_full("новый абзац"), "новый абзац")
check("cmd: новый параграф", match_command_full("новый параграф"), "новый абзац")

check("cmd: таб", match_command_full("таб"), "таб")
check("cmd: табуляция", match_command_full("табуляция"), "таб")

check("no cmd: привет", match_command_full("привет"), None)
check("no cmd: удалить", match_command_full("удалить"), None)
check("no cmd: слово", match_command_full("слово"), None)
check("no cmd: random text", match_command_full("я хочу есть"), None)
check("no cmd: partial", match_command_full("удалить сл"), None)
check("no cmd: embedded", match_command_full("пожалуйста удалить слово сейчас"), None)

# ===== COMMAND TRAILING MATCH =====
print("\n=== Command Trailing Match ===")

check("trail: only cmd", match_command_trailing("удалить слово"), ("", "удалить слово"))
check("trail: text + cmd", match_command_trailing("привет как дела удалить слово"),
      ("привет как дела", "удалить слово"))
check("trail: text + cmd dot", match_command_trailing("привет удалить слово."),
      ("привет", "удалить слово"))
check("trail: text + отменить", match_command_trailing("текст отменить"),
      ("текст", "отмена"))
check("trail: text + новая строка", match_command_trailing("что-то новая строка"),
      ("что-то", "новая строка"))
check("trail: multi word before", match_command_trailing("раз два три удалить всё"),
      ("раз два три", "удалить всё"))
check("trail: no cmd", match_command_trailing("привет как дела"),
      ("привет как дела", None))
check("trail: empty", match_command_trailing(""), ("", None))
check("trail: single word no match", match_command_trailing("привет"),
      ("привет", None))

# ===== DIFF ALGORITHM =====
print("\n=== Smart Diff ===")

check("diff: empty to text", compute_diff("", "Привет"), (0, "Привет"))
check("diff: append", compute_diff("Привет", "Привет как"), (0, " как"))
check("diff: append more", compute_diff("Привет как", "Привет как дела"),
      (0, " дела"))
check("diff: change suffix", compute_diff("Привет как", "Привет, как дела"),
      (4, ", как дела"))
check("diff: full replace", compute_diff("Здравствуйте", "Привет"),
      (12, "Привет"))
check("diff: no change", compute_diff("Тест", "Тест"), (0, ""))
check("diff: shorten", compute_diff("Привет мир", "Привет"), (4, ""))
check("diff: one char change", compute_diff("кот", "код"), (1, "д"))
check("diff: empty to empty", compute_diff("", ""), (0, ""))
check("diff: unicode", compute_diff("ёж", "ёжик"), (0, "ик"))

# ===== EDGE CASES =====
print("\n=== Edge Cases ===")

check("edge: cmd with extra spaces", match_command_full("  удалить  слово  "), "удалить слово")
check("edge: cmd whisper output Удалить слово,", match_command_full("Удалить слово,"), "удалить слово")
check("edge: cmd whisper output Удалить слово!", match_command_full("Удалить слово!"), "удалить слово")
check("edge: halluc substring in legit", is_hallucination("Я сказал спасибо за помощь"), False)
check("edge: halluc exact you", is_hallucination("you"), True)
check("edge: legit 'youtube'", is_hallucination("youtube"), True)  # contains "you"
check("edge: trailing cmd удали всё.", match_command_trailing("текст удали всё."),
      ("текст", "удалить всё"))

# whisper variations
check("whisper: Удалить Слово", match_command_full("Удалить Слово"), "удалить слово")
check("whisper: удалить слово...", match_command_full("удалить слово..."), "удалить слово")

# ===== SUMMARY =====
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
if failed == 0:
    print("All tests passed!")
else:
    print(f"FAILURES: {failed}")
    sys.exit(1)
