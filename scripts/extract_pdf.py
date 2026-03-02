from pathlib import Path
import pdfplumber

# ========= 1️⃣ 获取项目根目录 =========

project_root = Path(__file__).resolve().parent.parent

# ========= 2️⃣ 设置路径 =========

pdf_path = project_root / "pdf" / "covid.pdf"

output_dir = project_root / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "covid.txt"

# ========= 3️⃣ 提取文本 =========

all_text = ""

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            all_text += f"\n\n===== 第 {i+1} 页 =====\n\n"
            all_text += text

# ========= 4️⃣ 写入文件 =========

with open(output_path, "w", encoding="utf-8") as f:
    f.write(all_text)

print("✅ 提取完成")
print("输出路径：", output_path)