"""Сборка golden set: ручное ядро плюс синтетика по шаблонам.

Design document требует 300-500 размеченных тикетов с продоподобным
распределением (FAQ доминирует) и минимум 50 примерами на каждую high-risk
категорию. Набирать это руками нерационально, поэтому ядро (`golden_core.jsonl`)
написано вручную, а объём добирается шаблонами - ровно та «синтетика по
шаблонам реальных обращений», что описана в разделе «Методика оценки».

Разметка `expected_slugs` привязана к слоту, а не к шаблону: тему обращения
задаёт подставляемый фрагмент («какая гарантия на технику?»), а не обёртка
(«Подскажите, ...»). Привязка к шаблону давала бы неверную разметку и делала
метрику Recall@5 бессмысленной.

Честное ограничение: доля из публичных датасетов (по документу - около 40%)
здесь отсутствует, набор полностью синтетический. Метрики на нём показывают
поведение системы на ожидаемых формулировках, но не устойчивость к реальному
языку клиентов - сленгу, опечаткам, смешанным обращениям.

    python -m scripts.gen_golden_set --size 320
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
CORE_PATH = EVAL_DIR / "golden_core.jsonl"
OUT_PATH = EVAL_DIR / "golden_set.jsonl"

#: Продоподобное распределение: FAQ и статус заказа доминируют, high-risk редки.
#: Доли сознательно не выравнены - выравнивание исказило бы оценку recall.
DISTRIBUTION = {
    "faq": 0.32,
    "order_status": 0.26,
    "tech_issue": 0.22,
    "complaint": 0.10,
    "refund": 0.10,
}

#: Минимум примеров на high-risk категорию: ниже этого recall не измерим.
MIN_HIGH_RISK = 50

#: Обёртки. Тему не задают, поэтому документов за ними не закреплено.
TEMPLATES: dict[str, list[str]] = {
    "faq": [
        "Подскажите, {question}",
        "Добрый день! {question}",
        "Хотел уточнить: {question}",
        "{question} Заранее спасибо",
    ],
    "order_status": [
        "Где мой заказ {order}? {detail}",
        "Заказ {order} висит в одном статусе. {detail}",
        "Когда доставят заказ {order}? {detail}",
        "Не могу отследить посылку по заказу {order}. {detail}",
    ],
    "tech_issue": [
        "{problem} Помогите разобраться",
        "У меня {problem}",
        "{problem} Что делать?",
        "Второй день {problem}",
    ],
    "complaint": [
        "Пишу претензию: {grievance}",
        "{grievance} Считаю это недопустимым",
        "Жалоба: {grievance}",
        "{grievance} Требую разобраться",
    ],
    "refund": [
        "Хочу вернуть деньги: {reason}",
        "Оформите возврат средств, {reason}",
        "Прошу возврат денег за заказ {order}, {reason}",
        "Верните оплату, {reason}",
    ],
}

#: Тематические слоты: значение + документы базы знаний, которые обязаны
#: оказаться в top-k. Это и есть разметка для Recall@5.
TOPIC_SLOTS: dict[str, list[tuple[str, list[str]]]] = {
    "question": [
        ("какие сроки доставки в Екатеринбург?", ["delivery-terms"]),
        ("сколько стоит доставка курьером?", ["delivery-cost"]),
        ("можно ли оплатить через СБП?", ["payment-methods"]),
        ("какая гарантия на бытовую технику?", ["warranty-terms"]),
        ("как получить чек на заказ?", ["invoice-and-documents"]),
        ("можно ли забрать заказ в выходной?", ["pickup-point-storage"]),
        ("какие условия бесплатной доставки?", ["delivery-cost"]),
        ("как изменить номер телефона в аккаунте?", ["account-sms-not-received"]),
        ("можно ли вернуть товар без чека?", ["return-policy"]),
        ("сколько хранится заказ в пункте выдачи?", ["pickup-point-storage"]),
        ("как восстановить пароль?", ["account-password-reset"]),
        ("как оформить заказ на юридическое лицо?", ["invoice-and-documents"]),
        ("можно ли проверить товар при получении?", ["pickup-point-check"]),
        ("действует ли промокод на товары со скидкой?", ["promocode-rules"]),
    ],
    "problem": [
        ("приложение вылетает при открытии корзины", ["app-cart-empty", "app-crash-on-start"]),
        ("не приходит SMS с кодом подтверждения", ["account-sms-not-received"]),
        ("зависает экран оплаты", ["app-payment-screen-freeze"]),
        ("не работает поиск по каталогу", ["site-search-not-working"]),
        ("не могу войти в личный кабинет", ["account-sms-not-received"]),
        ("промокод не применяется, пишет ошибку", ["promocode-not-working"]),
        ("не приходят уведомления о заказе", ["app-push-notifications"]),
        ("деньги списались дважды за один заказ", ["payment-double-charge"]),
        ("корзина очищается сама при перезапуске", ["app-cart-empty"]),
        ("приложение зависает на экране загрузки", ["app-crash-on-start"]),
        ("оплата не прошла, а деньги списались", ["payment-failed"]),
        ("кнопка оплаты не нажимается", ["app-payment-screen-freeze"]),
    ],
    "grievance": [
        ("товар пришёл с трещиной на корпусе", ["damaged-on-delivery"]),
        ("курьер нахамил и не дождался проверки", ["damaged-on-delivery"]),
        ("прислали не тот цвет и отказались менять", ["return-defective"]),
        ("коробка была вскрыта, часть вложения отсутствует", ["order-partial-delivery"]),
        ("третье обращение остаётся без ответа", ["support-working-hours"]),
        ("заявленная комплектация не соответствует реальной", ["return-defective"]),
        ("доставку переносили четыре раза без предупреждения", ["delivery-terms"]),
        ("цена выросла уже после оформления", ["price-changed-after-order"]),
        ("товар явно был в употреблении", ["return-defective"]),
        ("часть заказа не приехала, а в статусе всё доставлено", ["order-partial-delivery"]),
    ],
    "reason": [
        ("товар не подошёл по размеру", ["return-policy"]),
        ("качество не соответствует описанию", ["return-defective"]),
        ("устройство не включается", ["return-defective"]),
        ("заказ отменён складом", ["order-cancel", "refund-timeline"]),
        ("духи оказались не тем ароматом", ["return-non-returnable"]),
        ("товар оказался бракованным", ["return-defective"]),
        ("доставили не тот товар", ["return-policy"]),
        ("деньги за отмену так и не пришли", ["refund-timeline"]),
        ("товар перестал работать через два дня", ["return-defective"]),
    ],
}

#: Нейтральные слоты: на тему не влияют, документов за ними нет.
FILLER_SLOTS: dict[str, list[str]] = {
    "order": ["1023456", "1042315", "998877", "1200341", "1150022", "1078654"],
    "detail": [
        "Оформлял пять дней назад.",
        "Оплата прошла, подтверждение пришло.",
        "В личном кабинете статус не обновляется.",
        "Перевозчик молчит.",
        "Обещали доставку вчера.",
        "Деньги списаны, заказ не двигается.",
        "Пункт выдачи говорит, что посылка не поступала.",
        "Курьер не выходит на связь.",
    ],
}

#: Хвосты обращения. Нужны не для красоты, а для комбинаторики: без них
#: шаблоны × слоты дают слишком мало уникальных текстов, и набор упирается
#: в потолок задолго до целевого объёма.
SUFFIXES = [
    "",
    " Спасибо!",
    " Жду ответа.",
    " Ответьте, пожалуйста, поскорее.",
    " Заказ оформлен через приложение.",
    " Обращаюсь второй раз.",
]

#: Документы, релевантные самой категории независимо от формулировки.
CATEGORY_SLUGS = {
    "order_status": ["order-tracking", "order-status-meaning"],
}


def _fill(template: str, rng: random.Random) -> tuple[str, list[str]]:
    text = template
    slugs: list[str] = []

    for slot, values in TOPIC_SLOTS.items():
        placeholder = "{" + slot + "}"
        if placeholder in text:
            value, value_slugs = rng.choice(values)
            text = text.replace(placeholder, value)
            slugs.extend(value_slugs)

    for slot, values in FILLER_SLOTS.items():
        placeholder = "{" + slot + "}"
        if placeholder in text:
            text = text.replace(placeholder, rng.choice(values))

    return (text + rng.choice(SUFFIXES)).strip(), slugs


def build(size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    core = [
        json.loads(line)
        for line in CORE_PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]

    counts = {
        category: sum(1 for item in core if item["category"] == category)
        for category in DISTRIBUTION
    }
    target = {
        category: max(
            round(size * share),
            MIN_HIGH_RISK if category in ("complaint", "refund") else 0,
        )
        for category, share in DISTRIBUTION.items()
    }

    generated: list[dict] = []
    seen: set[str] = {item["text"] for item in core}

    for category, needed in target.items():
        templates = TEMPLATES[category]
        attempts = 0
        while counts[category] < needed and attempts < needed * 50:
            attempts += 1
            text, slugs = _fill(rng.choice(templates), rng)
            if text in seen:
                continue
            seen.add(text)
            counts[category] += 1
            generated.append(
                {
                    "id": f"g{len(generated) + 1:04d}",
                    "text": text,
                    "category": category,
                    "expected_slugs": sorted(set(slugs + CATEGORY_SLUGS.get(category, []))),
                    "synthetic": True,
                }
            )

    return core + generated


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Сборка golden set")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260918, help="фиксирует воспроизводимость")
    args = parser.parse_args()

    items = build(args.size, args.seed)
    OUT_PATH.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
        encoding="utf-8",
    )

    by_category: dict[str, int] = {}
    for item in items:
        by_category[item["category"]] = by_category.get(item["category"], 0) + 1

    print(f"golden set: {len(items)} тикетов → {OUT_PATH}")
    for category, count in sorted(by_category.items(), key=lambda pair: -pair[1]):
        print(f"  {category:14} {count:4}")
    ambiguous = sum(1 for item in items if item.get("ambiguous"))
    labelled = sum(1 for item in items if item.get("expected_slugs"))
    print(f"  из них ambiguous: {ambiguous}; с разметкой документов: {labelled}")


if __name__ == "__main__":
    main()
