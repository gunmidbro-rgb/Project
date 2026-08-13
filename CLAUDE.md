# Project Overview
โปรเจกต์นี้คือ End-to-End Natural Language Processing (NLP) บน Amazon Reviews Dataset เริ่มตั้งแต่การทำความสะอาดข้อมูล (Data Cleaning), การแปลงข้อความ (Text Preprocessing & Feature Extraction) ไปจนถึงการสร้างและประเมินผลโมเดล NLP

# Tech Stack & Environment
- Python 3.10+
- Main Libraries: Pandas, NumPy, Scikit-learn, spaCy, Transformers (Hugging Face)
- Environment: Virtual environment (`venv`)

# Folder Structure
- `data/`: โฟลเดอร์สำหรับเก็บข้อมูลดิบ (Raw data), ข้อมูลที่ถูกแตกไฟล์ออกมา, และข้อมูลที่ผ่านการทำความสะอาดแล้ว (Cleaned/Processed data)
- `notebooks/`: ใช้สร้าง Jupyter Notebooks สำหรับทดลองรันข้อมูล, ทำ EDA (Exploratory Data Analysis) และทดสอบแนวคิดก่อนนำไปเขียนเป็นสคริปต์จริง
- `src/`: โฟลเดอร์หลักที่เก็บฟังก์ชันการทำงานหลัก (Source Code), สคริปต์ Clean ข้อมูล, โมดูลแปลง Feature และ Pipeline สำหรับการเทรนโมเดล

# Common Commands
- Activate Environment: `source venv/bin/activate` (Mac/Linux) หรือ `venv\Scripts\activate` (Windows)
- Install Dependencies: `pip install -r requirements.txt`
- Run Notebook: `jupyter notebook`
- Run Main Scripts: `python src/<script_name>.py`

# Coding Guidelines
1. **Modular Structure:**
   - รหัสโค้ดสำหรับการทดลอง ให้ทำใน `notebooks/`
   - ฟังก์ชันที่ใช้งานซ้ำได้บ่อยๆ (เช่น การ Clean Text, การตัดคำ) ให้ย้ายจาก Notebook มาเขียนเป็นฟังก์ชันไว้ใน `src/` (เช่น `src/utils.py` หรือ `src/cleaner.py`)
2. **Code Style:**
   - เขียนโค้ดตามมาตรฐาน PEP8
   - ใส่ Type Hints และ Docstrings สั้นๆ 
   - กำหนด `random_state=42` เสมอเมื่อต้องสุ่มข้อมูล เพื่อให้ผลลัพธ์รันซ้ำได้เหมือนเดิม (Reproducibility)

# Data & Memory Rules (IMPORTANT)
- **Do NOT commit data:** ห้าม Push ไฟล์ในโฟลเดอร์ `data/` ขึ้น Git เด็ดขาด (ต้องมั่นใจว่าเพิ่ม `data/` ไว้ใน `.gitignore` แล้ว)
- **Memory Management:** หากจัดการกับไฟล์ขนาดใหญ่ใน `data/` ให้ใช้วิธีอ่านข้อมูลแบบเป็น Chunk (`chunksize`) เพื่อไม่ให้ RAM ของเครื่องเต็ม
- **Path Handling:** ใช้ Relative Path เสมอ เช่น `data/raw/reviews.csv` ห้ามใช้ Absolute Path ส่วนตัวของเครื่องเด็ดขาด

# Workflow
ก่อนแก้โค้ดให้ทำดังนี้:
1. อ่านไฟล์ที่เกี่ยวข้องก่อน
2. อธิบายแผนที่จะแก้ก่อนสั้นๆ
3. แก้แค่เฉพาะไฟล์ที่จำเป็นและสั่ง
4. ตรวจสอบว่าโค้ดไม่กระทบส่วนอื่น
5. สรุปสิ่งที่แก้มาโดยสังเขป