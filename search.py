"""Food Search: семантический поиск картинок блюд по текстовому запросу.

Модель: OpenCLIP ViT-B/32 (текст — OpenAI) + опционально визуальные веса Food101.
Запуск:  python search.py [--variant food101|openai] [--rebuild] [--top-k 3] [--no-show]
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
import warnings

import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
# Argos и его токенизатор по умолчанию пишут в консоль десятки служебных строк на каждый запрос.
for _name in ("argostranslate", "stanza"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Пути считаем от папки скрипта, а не от текущей директории: можно запускать откуда угодно.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
TRANSLATION_CACHE_FILE = os.path.join(BASE_DIR, "translations_cache.json")

# Для каждого набора весов свой индекс: эмбеддинги от разных весов нельзя смешивать.
# Старый food_clip_index.pkl остаётся индексом для обычного OpenAI CLIP.
INDEX_FILES = {
    "openai": os.path.join(BASE_DIR, "food_clip_index.pkl"),
    "food101": os.path.join(BASE_DIR, "food_clip_index_food101.pkl"),
}

MODEL_NAME = "ViT-B-32"
FOOD_REPO = "tanganke/clip-vit-base-patch32_food101"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
BATCH_SIZE = 64
NUM_WORKERS = 4


# ------------------------------------------------------------ загрузка весов

def hf_vision_to_open_clip(sd: dict) -> dict:
    """Переименовывает ключи визуальной части из формата HuggingFace (transformers) в формат open_clip.

    Почему нужна конвертация: у HF и open_clip одни и те же слои называются по-разному
    (например, layers.0.self_attn.q_proj против resblocks.0.attn.in_proj_weight),
    а q/k/v в open_clip хранятся одним склеенным тензором.
    Бросает KeyError, если формат чекпоинта оказался другим.
    """
    p = "vision_model."
    out = {
        "class_embedding": sd[p + "embeddings.class_embedding"],
        "conv1.weight": sd[p + "embeddings.patch_embedding.weight"],
        "positional_embedding": sd[p + "embeddings.position_embedding.weight"],
        "ln_pre.weight": sd[p + "pre_layrnorm.weight"],
        "ln_pre.bias": sd[p + "pre_layrnorm.bias"],
        "ln_post.weight": sd[p + "post_layernorm.weight"],
        "ln_post.bias": sd[p + "post_layernorm.bias"],
    }

    layers = 0
    while f"{p}encoder.layers.{layers}.layer_norm1.weight" in sd:
        layers += 1
    if layers == 0:
        raise KeyError("в чекпоинте не найдены слои encoder.layers")

    for i in range(layers):
        src = f"{p}encoder.layers.{i}."
        dst = f"transformer.resblocks.{i}."
        for part in ("weight", "bias"):
            out[dst + f"ln_1.{part}"] = sd[src + f"layer_norm1.{part}"]
            out[dst + f"ln_2.{part}"] = sd[src + f"layer_norm2.{part}"]
            out[dst + f"attn.in_proj_{part}"] = torch.cat(
                [sd[src + f"self_attn.{x}_proj.{part}"] for x in ("q", "k", "v")], dim=0
            )
            out[dst + f"attn.out_proj.{part}"] = sd[src + f"self_attn.out_proj.{part}"]
            out[dst + f"mlp.c_fc.{part}"] = sd[src + f"mlp.fc1.{part}"]
            out[dst + f"mlp.c_proj.{part}"] = sd[src + f"mlp.fc2.{part}"]

    # Проекция в общее пространство с текстом. Если её нет в чекпоинте — остаётся от OpenAI.
    if "visual_projection.weight" in sd:
        out["proj"] = sd["visual_projection.weight"].t().contiguous()
    return out


def try_load_food_weights(model) -> bool:
    """Загружает веса Food101 в model.visual. True — только если загрузились ВСЕ слои.

    Сначала всё проверяем и только потом загружаем: load_state_dict(strict=False)
    молча загрузил бы часть ключей и оставил модель «наполовину дообученной».
    """
    import safetensors.torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=FOOD_REPO, filename="model.safetensors")
    raw = safetensors.torch.load_file(path, device="cpu")
    converted = hf_vision_to_open_clip(raw)

    target = model.visual.state_dict()
    missing = set(target) - set(converted) - {"proj"}
    unexpected = set(converted) - set(target)
    bad_shapes = [k for k in converted if k in target and converted[k].shape != target[k].shape]

    if missing or unexpected or bad_shapes:
        print(
            f"   ⚠️ Веса Food101 не подходят: не хватает {len(missing)}, "
            f"лишних {len(unexpected)}, не совпало по форме {len(bad_shapes)}."
        )
        return False

    converted = {k: v.to(target[k].dtype) for k, v in converted.items()}
    model.visual.load_state_dict(converted, strict=False)
    return True


def load_model(requested_variant: str, device: str):
    """Возвращает (model, preprocess, tokenizer, фактический_вариант)."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained="openai")
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    variant = "openai"

    if requested_variant == "food101":
        print("   Загрузка визуальных весов Food101...")
        try:
            if try_load_food_weights(model):
                variant = "food101"
        except Exception as e:
            print(f"   ⚠️ Не удалось загрузить веса Food101: {e}")
        if variant != "food101":
            print("   ➡️ Продолжаю на обычном OpenAI CLIP.")

    model.to(device).eval()
    return model, preprocess, tokenizer, variant


# -------------------------------------------------------------------- индекс

class ImageDataset(torch.utils.data.Dataset):
    """Читает и препроцессит картинки в фоновых процессах, пока GPU считает прошлый батч."""

    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as img:
                return i, self.preprocess(img.convert("RGB"))
        except Exception:
            return i, None  # битый файл пропускаем, а не роняем весь индекс


def collate_skip_broken(batch):
    batch = [(i, t) for i, t in batch if t is not None]
    if not batch:
        return None, None
    return [i for i, _ in batch], torch.stack([t for _, t in batch])


def list_images(folder: str) -> list:
    found = []
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                found.append(os.path.join(root, name))
    found.sort()
    return found


def build_index(model, preprocess, device, variant):
    """Кодирует все картинки батчами и сохраняет индекс. Пути хранятся относительно images/."""
    print("\n📁 Сканирование папки...")
    if not os.path.isdir(IMAGE_DIR):
        print(f"❌ Папка '{IMAGE_DIR}' не найдена.")
        return None
    all_images = list_images(IMAGE_DIR)
    if not all_images:
        print(f"❌ В папке '{IMAGE_DIR}' нет изображений.")
        return None

    print(f"Найдено {len(all_images)} изображений. Кодирование батчами по {BATCH_SIZE}...")
    loader = torch.utils.data.DataLoader(
        ImageDataset(all_images, preprocess),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_skip_broken,
    )

    rel_paths, chunks, done = [], [], 0
    use_amp = device == "cuda"
    with torch.no_grad():
        for indices, batch in loader:
            done += BATCH_SIZE
            if batch is None:
                continue
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                feats = model.encode_image(batch.to(device))
            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.cpu().numpy())
            rel_paths.extend(os.path.relpath(all_images[i], IMAGE_DIR) for i in indices)
            if (len(rel_paths) // BATCH_SIZE) % 10 == 0:
                print(f"  Обработано: {min(done, len(all_images))}/{len(all_images)}")

    if not rel_paths:
        print("❌ Не удалось закодировать ни одной картинки.")
        return None

    features = np.concatenate(chunks).astype("float32")
    with open(INDEX_FILES[variant], "wb") as f:
        pickle.dump({"paths": rel_paths, "features": features, "variant": variant}, f)
    print(f"✅ Индекс сохранён: {len(rel_paths)} картинок ({os.path.basename(INDEX_FILES[variant])})")
    return rel_paths, features


def load_index(variant):
    path = INDEX_FILES[variant]
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        raw_paths, features = data["paths"], data["features"]
    except Exception as e:
        print(f"⚠️ Индекс {os.path.basename(path)} повреждён ({e}), пересоберу.")
        return None

    # Новый индекс (с ключом 'variant') уже хранит пути относительно images/.
    # Старый хранил пути вроде 'images\\x.jpg' относительно папки запуска —
    # их приводим к тому же виду, чтобы скрипт не зависел от текущей директории.
    new_format = "variant" in data
    rel_paths = []
    for p in raw_paths:
        if new_format:
            rel_paths.append(p)
            continue
        absolute = p if os.path.isabs(p) else os.path.normpath(os.path.join(BASE_DIR, p))
        rel_paths.append(os.path.relpath(absolute, IMAGE_DIR))

    if len(rel_paths) != len(features):
        print("⚠️ Индекс несогласован (число путей != числу векторов), пересоберу.")
        return None
    print(f"💾 Индекс загружен: {len(rel_paths)} картинок")
    return rel_paths, features


# -------------------------------------------------------------------- поиск

class Translator:
    """Перевод ru -> en. Порядок: кэш -> Argos (офлайн) -> Google (запасной) -> исходный текст.

    Почему Argos первым: он работает локально на CPU, без лимитов и интернета
    (интернет нужен один раз, чтобы скачать языковой пакет).
    """

    def __init__(self):
        self._argos = None          # объект перевода Argos, пока не загружен — None
        self._argos_failed = False  # если Argos не взлетел, больше не пытаемся
        self._google = None
        self._cache = {}
        try:
            with open(TRANSLATION_CACHE_FILE, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        except (OSError, ValueError):
            pass

    def _load_argos(self):
        """Находит русско-английский пакет Argos, а при первом запуске скачивает его."""
        if self._argos is not None or self._argos_failed:
            return self._argos
        try:
            import argostranslate.package as package
            import argostranslate.translate as translate

            def find_installed():
                langs = {lang.code: lang for lang in translate.get_installed_languages()}
                if "ru" in langs and "en" in langs:
                    return langs["ru"].get_translation(langs["en"])
                return None

            self._argos = find_installed()
            if self._argos is None:
                print("📥 Скачиваю русско-английский пакет Argos (один раз, около 100 МБ)...")
                package.update_package_index()
                pkg = next(
                    (p for p in package.get_available_packages()
                     if p.from_code == "ru" and p.to_code == "en"),
                    None,
                )
                if pkg is None:
                    raise RuntimeError("пакет ru->en не найден в индексе Argos")
                package.install_from_path(pkg.download())
                self._argos = find_installed()
            if self._argos is None:
                raise RuntimeError("пакет скачан, но не появился в списке установленных")
        except Exception as e:
            print(f"⚠️ Argos недоступен: {e}")
            self._argos_failed = True
        return self._argos

    def _google_translate(self, text: str):
        try:
            if self._google is None:
                from deep_translator import GoogleTranslator
                self._google = GoogleTranslator(source="auto", target="en")
            return self._google.translate(text)
        except Exception as e:
            print(f"⚠️ Google Translate не сработал: {e}")
            return None

    def to_english(self, text: str) -> str:
        if text.isascii():  # уже английский — перевод не нужен
            return text
        key = text.strip().lower()
        if key in self._cache:
            return self._cache[key]

        result = None
        argos = self._load_argos()
        if argos is not None:
            try:
                result = argos.translate(text)
            except Exception as e:
                print(f"⚠️ Ошибка перевода Argos: {e}")
        if not result:
            result = self._google_translate(text)
        if not result or not result.strip():
            print("⚠️ Перевести не удалось, использую исходный текст.")
            return text

        result = result.strip()
        self._cache[key] = result
        try:
            with open(TRANSLATION_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=1)
        except OSError:
            pass
        return result


def search(query, top_k, model, tokenizer, device, translator, feats_t, rel_paths):
    """Возвращает (результаты, тайминги). Сходство считается на GPU одной матричной операцией."""
    t0 = time.perf_counter()
    english = translator.to_english(query)
    t1 = time.perf_counter()

    tokens = tokenizer([english]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(tokens).float()
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        sims = feats_t @ text_feat.squeeze(0)  # косинусное сходство: векторы уже нормированы
        scores, idx = torch.topk(sims, min(top_k, sims.numel()))
    if device == "cuda":
        torch.cuda.synchronize()  # без этого замер времени покажет только постановку задачи в очередь
    t2 = time.perf_counter()

    results = [
        {"rel_path": rel_paths[i], "path": os.path.join(IMAGE_DIR, rel_paths[i]), "score": float(s)}
        for s, i in zip(scores.tolist(), idx.tolist())
    ]
    return english, results, {"translate": t1 - t0, "search": t2 - t1, "total": t2 - t0}


# --------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description="Семантический поиск картинок еды")
    parser.add_argument("--variant", choices=("food101", "openai"), default="food101",
                        help="какие визуальные веса использовать (по умолчанию food101 с откатом на openai)")
    parser.add_argument("--rebuild", action="store_true", help="пересобрать индекс")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-show", action="store_true", help="не открывать картинку")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Устройство: {device}")
    print("📥 Инициализация модели...")
    try:
        model, preprocess, tokenizer, variant = load_model(args.variant, device)
    except Exception as e:
        print(f"❌ Не удалось загрузить модель: {e}")
        return 1
    label = "Vision: Food101 (дообучено)" if variant == "food101" else "Vision: OpenAI CLIP (без дообучения)"
    print(f"✅ Модель готова (Text: OpenAI, {label})")

    loaded = None if args.rebuild else load_index(variant)
    if loaded is None:
        loaded = build_index(model, preprocess, device, variant)
    if loaded is None:
        return 1
    rel_paths, feats = loaded

    # Матрицу векторов кладём на устройство один раз: 9489 x 512 float32 ≈ 19 МБ.
    feats_t = torch.from_numpy(feats).to(device)
    translator = Translator()

    print("\n" + "=" * 50)
    print(f"🍔 Food Search ({'OpenCLIP + Food101' if variant == 'food101' else 'OpenCLIP'})")
    print("Для выхода: exit / quit / выход")
    print("=" * 50)

    while True:
        try:
            query = input("\nЗапрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in ("exit", "quit", "выход"):
            break
        if not query:
            continue

        try:
            english, results, timing = search(
                query, args.top_k, model, tokenizer, device, translator, feats_t, rel_paths
            )
        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")
            continue

        if english != query:
            print(f"🌐 Перевод: {query} → {english}")
        print(
            f"⏱️ Всего {timing['total']:.3f} с "
            f"(перевод {timing['translate']:.3f} с, кодирование текста и поиск {timing['search']:.3f} с)"
        )
        for n, r in enumerate(results, 1):
            print(f"  {n}. [{r['score']:.3f}] {r['rel_path']}")

        if results and not args.no_show:
            try:
                Image.open(results[0]["path"]).show()
            except Exception as e:
                print(f"⚠️ Не удалось показать картинку: {e}")
    return 0


if __name__ == "__main__":
    # Защита нужна на Windows: DataLoader с num_workers запускает дочерние процессы,
    # которые заново импортируют этот файл.
    sys.exit(main())
