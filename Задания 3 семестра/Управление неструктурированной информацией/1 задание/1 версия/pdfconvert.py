import os
import fitz
from PIL import Image, ImageEnhance
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

INPUT_DIR = "input"
OUTPUT_DIR = "output"


def to_grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L")


def to_grayscale_high_contrast(image: Image.Image) -> Image.Image:
    """Оттенки серого с высоким контрастом — почти ч/б эффект, без рваных пикселей."""
    img = image.convert("L")   # grayscale

    # Усиливаем контраст
    img = ImageEnhance.Contrast(img).enhance(2.0)

    # Усиливаем резкость
    img = ImageEnhance.Sharpness(img).enhance(1.5)

    # Лёгкое осветление (фон станет белее)
    img = ImageEnhance.Brightness(img).enhance(1.10)

    # Лёгкое затемнение тёмных участков
    img = img.point(lambda x: max(0, min(255, int(x * 1.08))))

    return img


def to_binary(image):
    return image.convert("L").point(lambda x: 0 if x < 128 else 255, "1")


def is_grayscale(image: Image.Image) -> bool:
    img = image.convert("RGB")
    for r, g, b in img.getdata():
        if r != g or g != b:
            return False
    return True


def images_to_pdf(images, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    for img in images:
        img = img.resize((int(width), int(height)), Image.LANCZOS)
        temp = "temp_page.png"
        img.save(temp)
        c.drawImage(temp, 0, 0, width, height)
        c.showPage()
        os.remove(temp)

    c.save()


def check_pdf_size(path):
    doc = fitz.open(path)
    w, h = doc[0].rect.width, doc[0].rect.height
    print(f"Размер PDF: {w:.2f} x {h:.2f} points")
    if abs(w - 595) < 3 and abs(h - 842) < 3:
        print("✔ Формат A4")
    else:
        print("❗ НЕ A4")


def process_pdf(input_pdf, output_pdf, mode):
    print(f"\nОбрабатываю: {input_pdf}")

    doc = fitz.open(input_pdf)
    images = []

    for num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        print(f" → Страница {num}: ", end="")

        # уже ч/б?
        if is_grayscale(img):
            print("уже ч/б — лёгкая обработка")
            processed = to_grayscale_high_contrast(img)
            images.append(processed)
            continue

        print("цветная — конвертирую")

        if mode == "binary":
            processed = to_binary(img)
        elif mode == "grayscale":
            processed = to_grayscale(img)
        elif mode == "highc":
            processed = to_grayscale_high_contrast(img)
        else:
            processed = to_grayscale(img)

        images.append(processed)

    images_to_pdf(images, output_pdf)

    print(f"Готово → {output_pdf}")
    check_pdf_size(output_pdf)


def main():
    print("Выберите режим:")
    print("1 — Чистая ч/б (binary)")
    print("2 — Обычный grayscale")
    print("3 — High Contrast Grayscale (рекомендуется)")

    choice = input("Введите 1/2/3: ")

    if choice == "1":
        mode = "binary"
    elif choice == "2":
        mode = "grayscale"
    elif choice == "3":
        mode = "highc"
    else:
        print("Неверный ввод")
        return

    if not os.path.exists(INPUT_DIR):
        print("Нет папки input")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    print(f"\nНайдено PDF: {len(pdf_files)}")

    for pdf in pdf_files:
        in_path = os.path.join(INPUT_DIR, pdf)
        out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(pdf)[0]}_{mode}.pdf")
        process_pdf(in_path, out_path, mode)

if __name__ == "__main__":
    main()

    # Автоматическая выгрузка проекта на GitHub
    from upload_module import upload_to_github
    upload_to_github()

