import streamlit as st
from google import genai
from google.genai.errors import APIError
import pandas as pd
import json
import re
import time
from typing import List, Dict, Any, Optional

# --- Константы и Настройки ---
MODEL_NAME = "gemini-2.5-flash"

# JSON-схема для структурированного позиционного и тонального анализа (Шаг 4)
# Примечание: Мы по-прежнему просим LLM вернуть до 3 наиболее заметных брендов, 
# так как это обеспечивает лучшую точность ранжирования, но 3-е место получает базовый балл для всех последующих.
SOV_ANALYSIS_SCHEMA = {
    "type": "ARRAY",
    "description": "A ranked list of up to 3 brands from the provided competitor list that are clearly recommended, ordered by their prominence/rank (most prominent/first mention is index 0). Each entry must include the brand name and the associated sentiment.",
    "items": {
        "type": "OBJECT",
        "properties": {
            "brandName": {
                "type": "STRING",
                "description": "The brand name, exactly as provided in the 'СПИСОК_БРЕНДОВ'."
            },
            "sentiment": {
                "type": "STRING",
                "enum": ["Positive", "Neutral", "Negative"],
                "description": "The sentiment associated with the brand's mention in the text (Positive, Neutral, or Negative)."
            }
        },
        "required": ["brandName", "sentiment"],
        "propertyOrdering": ["brandName", "sentiment"]
    }
}

# --- Функции Взаимодействия с API (с Обработкой Ошибок и Повторами) ---

def generate_content_with_retry(
    client: genai.Client,
    prompt: str,
    system_instruction: Optional[str] = None,
    max_retries: int = 3,
    json_output: bool = False,
    response_schema: Optional[Dict[str, Any]] = None
) -> str | None:
    """
    Выполняет вызов Gemini API с обработкой исключений и экспоненциальной задержкой, 
    с поддержкой структурированного JSON-вывода.
    """
    for attempt in range(max_retries):
        try:
            # Настройка конфигурации генерации
            config_params = {}
            if json_output:
                config_params["response_mime_type"] = "application/json"
                # Use the complex schema if provided, otherwise a simple array of strings
                config_params["response_schema"] = response_schema if response_schema else {"type": "ARRAY", "items": {"type": "STRING"}}
            
            if system_instruction:
                config_params["system_instruction"] = system_instruction

            # Вызов API
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=genai.types.GenerateContentConfig(**config_params)
            )

            # Извлечение текста
            if response.candidates and response.candidates[0].content:
                return response.candidates[0].content.parts[0].text
            
            st.warning(f"Gemini вернул пустой ответ на попытке {attempt + 1}.")
            return None

        except APIError as e:
            st.error(f"Ошибка API (Попытка {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                st.warning(f"Ожидание {wait_time} секунд перед повторной попыткой...")
                time.sleep(wait_time)
            else:
                return None
        except Exception as e:
            st.error(f"Непредвиденная ошибка (Попытка {attempt + 1}/{max_retries}): {e}")
            return None
    return None

# --- Инициализация Состояния Streamlit ---

if 'step' not in st.session_state:
    st.session_state.step = 1
# Инициализация brand и industry, чтобы избежать AttributeError
if 'brand' not in st.session_state: 
    st.session_state.brand = ''
if 'industry' not in st.session_state: 
    st.session_state.industry = ''
    
if 'user_queries' not in st.session_state:
    st.session_state.user_queries = ""
if 'competitors' not in st.session_state:
    st.session_state.competitors = ""
if 'results' not in st.session_state:
    st.session_state.results = None
if 'client' not in st.session_state:
    st.session_state.client = None
if 'analysis_details' not in st.session_state:
    st.session_state.analysis_details = [] # Хранение детальных результатов
if 'raw_responses' not in st.session_state:
    st.session_state.raw_responses = [] # Хранение сырых ответов Gemini

# --- UI и Логика Приложения ---

st.set_page_config(
    page_title="AI Share of Voice (SoV) Analyzer MVP",
    layout="centered"
)

st.title("🗣️ AI Share of Voice (SoV) Анализатор (v3.1 - Все Упоминания)")
st.markdown("Измерьте, как часто Gemini рекомендует ваш бренд по сравнению с конкурентами, используя **структурированный LLM-анализ** для точного сопоставления.")

# --- Шаг 1: Ввод Данных ---

st.header("Шаг 1: Ввод Настроек")
st.info("Пожалуйста, заполните поля для начала анализа.")

with st.expander("Конфигурация", expanded=True):
    
    brand = st.text_input(
        "Ваш Бренд (YOUR_BRAND_NAME)", 
        # Используем значение из session_state для сохранения после reruns
        value=st.session_state.brand if st.session_state.brand else 'AI-SaaS Tracker Pro',
        help="Название вашего бренда, который вы отслеживаете."
    )
    industry = st.text_area(
        "Описание Индустрии (INDUSTRY_DESCRIPTION)", 
        # Используем значение из session_state для сохранения после reruns
        value=st.session_state.industry if st.session_state.industry else 'Инструменты для аналитики AI-решений и отслеживания метрик SaaS.',
        help="Подробное описание вашей категории продукта/рынка."
    )

    if st.button("Сохранить Настройки и Перейти к Шагу 2"):
        
        # --- Инициализация клиента ---
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("Ошибка: Ключ 'GEMINI_API_KEY' не найден в конфигурации.")
            pass
        else:
            api_key_to_use = st.secrets["GEMINI_API_KEY"]
            
            if brand and industry:
                try:
                    # Инициализируем клиент
                    st.session_state.client = genai.Client(api_key=api_key_to_use)
                    # Обновляем session_state после успешного ввода
                    st.session_state.brand = brand
                    st.session_state.industry = industry
                    st.session_state.step = 2 # Переход к Шагу 2 (Генерация Запросов)
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка инициализации клиента: {e}. Проверьте доступность API.")
            else:
                st.error("Пожалуйста, заполните поля 'Бренд' и 'Индустрия'.")

if st.session_state.step >= 2:
    st.divider()

    # --- Шаг 2: Генерация Рекомендательных Запросов ---

    st.header("Шаг 2: Генерация Запросов")
    st.markdown("Сгенерируйте **5** типовых запросов на прямую рекомендацию.")
    
    if st.button("Сгенерировать Рекомендательные Запросы", disabled=st.session_state.step != 2):
        if st.session_state.client:
            with st.spinner("Gemini генерирует запросы..."):
                prompt = (
                    f"На основе бренда '{st.session_state.brand}' и описания индустрии '{st.session_state.industry}', "
                    f"сгенерируй 5 наиболее распространенных запросов, ищущих прямую рекомендацию или сравнение (например, "
                    f"'лучший [категория]', 'посоветуй [категорию]'). Выведи только JSON-список строк."
                )
                
                json_response = generate_content_with_retry(
                    st.session_state.client, 
                    prompt, 
                    json_output=True
                )
                
                if json_response:
                    try:
                        queries = json.loads(json_response)
                        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                            st.session_state.user_queries = "\n".join(queries)
                            st.session_state.step = 3 # Переход к Шагу 3 (Ввод Конкурентов)
                            st.success("Запросы сгенерированы! Перейдите к Шагу 3.")
                        else:
                            st.error("Ошибка парсинга: Gemini не вернул корректный JSON-список строк.")
                    except json.JSONDecodeError:
                        st.error(f"Ошибка: Не удалось декодировать JSON. Ответ Gemini: {json_response[:200]}...")
                else:
                    st.error("Не удалось сгенерировать запросы.")

    if st.session_state.step >= 3:
        st.subheader("Финальные Запросы (Отредактируйте при необходимости):")
        st.session_state.user_queries = st.text_area(
            "Список запросов (один запрос на строку):", 
            value=st.session_state.user_queries,
            height=150
        )
        final_queries = [q.strip() for q in st.session_state.user_queries.split('\n') if q.strip()]
        
        st.caption(f"Будет использовано запросов: {len(final_queries)}")
    
    if st.session_state.step >= 3:
        st.divider()

        # --- Шаг 3: Ввод Конкурентов и Сбор Ответов Gemini ---
        
        st.header("Шаг 3: Ввод Конкурентов и Сбор Ответов")
        st.info("Введите список брендов, которые будут отслеживаться, и соберите ответы Gemini.")
        
        st.subheader("Список Отслеживаемых Брендов (через запятую):")
        st.session_state.competitors = st.text_area(
            "Список Брендов:", 
            value=st.session_state.competitors if st.session_state.competitors else st.session_state.brand,
            height=100,
            help="Убедитесь, что ваш бренд включен. Бренды должны быть разделены запятыми."
        )
        
        final_competitors = [c.strip() for c in st.session_state.competitors.split(',') if c.strip()]
        st.caption(f"Будет отслеживаться брендов: {len(final_competitors)}")


        if st.button("Получить Ответы Gemini", disabled=st.session_state.step != 3):
            if not final_queries:
                st.error("Убедитесь, что запросы заполнены в Шаге 2.")
            elif not final_competitors:
                 st.error("Убедитесь, что список отслеживаемых брендов заполнен.")
            elif st.session_state.client:
                st.session_state.raw_responses = [] # Сброс
                N = len(final_queries)
                progress_bar = st.progress(0, text="Идет получение ответов...")

                for i, query in enumerate(final_queries):
                    progress_value = (i + 1) / N
                    progress_bar.progress(progress_value, text=f"Получение ответа на запрос {i+1}/{N}")
                    
                    answer_text = generate_content_with_retry(
                        st.session_state.client, 
                        prompt=query, 
                        max_retries=2
                    )
                    
                    if answer_text:
                        st.session_state.raw_responses.append({'query': query, 'answer': answer_text})
                    else:
                        st.session_state.raw_responses.append({'query': query, 'answer': "Ошибка получения ответа API"})
                    
                progress_bar.progress(1.0, text="Сбор ответов завершен!")
                st.success(f"Собрано {len(st.session_state.raw_responses)} ответов. Перейдите к Шагу 4 для анализа.")
                st.session_state.step = 4 # Переход к Шагу 4 (Анализ)
                st.rerun()


if st.session_state.step >= 4:
    st.divider()
    
    # --- Шаг 4: Структурированный Анализ и Расчет AI SoV ---

    st.header("Шаг 4: Структурированный Анализ AI SoV (Позиция и Тональность)")
    st.info(f"Нажмите, чтобы проанализировать {len(st.session_state.raw_responses)} сырых ответов Gemini и рассчитать Share of Voice.")
    
    # --- Отображение сырых ответов (Данные для Шага 4) ---
    if st.session_state.raw_responses:
        st.subheader("Данные для Анализа (Сырые Ответы из Шага 3)")
        st.caption("Проверьте эти ответы. Анализ LLM будет проведен на основе этого текста.")
        for i, item in enumerate(st.session_state.raw_responses):
            with st.expander(f"Ответ {i+1}: {item['query'][:60]}..."):
                st.code(item['answer'], language='markdown')


    if st.button("Провести Структурированный Анализ и Расчет SoV", disabled=st.session_state.step != 4 or not st.session_state.raw_responses):
        if not final_competitors:
            st.error("Убедитесь, что конкуренты заполнены в Шаге 3.")
        elif st.session_state.client and st.session_state.raw_responses:
            
            # --- КОНСТАНТЫ СЧЕТА И МНОЖИТЕЛЕЙ ---
            # Базовый позиционный балл (Position Score)
            # 3-е место и все последующие получают базовый балл 1.0
            POSITION_SCORES = {
                0: 3.0, # 1st place
                1: 2.0, # 2nd place
                2: 1.0, # 3rd place и последующие
            }
            # Тональные множители (Sentiment Multipliers)
            SENTIMENT_MULTIPLIERS = {
                "Positive": 1.5,
                "Neutral": 1.0,
                "Negative": 0.0
            }
            # -------------------------------------------
            
            # Инициализация счетчиков
            brand_scores: Dict[str, float] = {brand.strip(): 0.0 for brand in final_competitors}
            total_tracked_score = 0.0 # Общий взвешенный счет всех упоминаний
            
            st.session_state.analysis_details = [] # Сброс и инициализация детального отчета
            
            N = len(st.session_state.raw_responses)
            TotalSteps = N 
            progress_bar = st.progress(0, text="Идет структурированный анализ...")

            for i, item in enumerate(st.session_state.raw_responses):
                query = item['query']
                answer_text = item['answer']
                
                # Пропускаем ответы с ошибками
                if answer_text == "Ошибка получения ответа API":
                    st.session_state.analysis_details.append({
                        'Запрос': query,
                        'Ответ Gemini': answer_text,
                        'Анализ (Позиция, Тональность, Счет)': "Ошибка",
                        'Общий Счет Запроса': 0.0
                    })
                    continue

                # Обновление прогресса
                progress_value = (i + 1) / TotalSteps
                progress_bar.progress(progress_value, text=f"Анализ упоминаний для запроса {i+1}/{N}")

                # 1. Структурированный анализ упоминаний брендов (LLM-анализ)
                system_instruction_analysis = (
                    "Вы — высокоточный движок позиционного и тонального анализа сущностей. "
                    "Внимательно проанализируйте весь предоставленный 'ТЕКСТ_ДЛЯ_АНАЛИЗА' (сырой ответ Gemini). "
                    "Ваша задача — определить, какие из брендов из 'СПИСОК_БРЕНДОВ' (включая собственный) являются наиболее рекомендуемыми или "
                    "наиболее заметными в этом тексте, и вернуть их в порядке убывания важности/ранга (максимум 3). "
                    "Для каждого бренда также определите тональность упоминания (Positive, Neutral, или Negative). "
                    "Используйте названия брендов СТРОГО из 'СПИСОК_БРЕНДОВ'. Выведите ТОЛЬКО JSON-объект, следуя предоставленной схеме. Не выводите другой текст."
                )
                
                analysis_prompt = (
                    f"ТЕКСТ_ДЛЯ_АНАЛИЗА: '''{answer_text}'''\n\n"
                    f"СПИСОК_БРЕНДОВ: {final_competitors}"
                )
                
                json_analysis_response = generate_content_with_retry(
                    st.session_state.client,
                    analysis_prompt,
                    system_instruction=system_instruction_analysis,
                    json_output=True,
                    response_schema=SOV_ANALYSIS_SCHEMA 
                )
                
                current_query_score = 0.0
                detected_brands_details = [] # [{'brandName': 'X', 'sentiment': 'Y', 'score': Z}]
                
                if json_analysis_response:
                    try:
                        ranked_brands_data = json.loads(json_analysis_response)
                        
                        if isinstance(ranked_brands_data, list):
                            for rank, brand_entry in enumerate(ranked_brands_data):
                                
                                brand_name_ranked = brand_entry.get('brandName', '').strip()
                                sentiment = brand_entry.get('sentiment', 'Neutral').strip()
                                
                                # 1. Определяем базовый позиционный балл
                                # Если ранг >= 2 (3-е место или ниже), используем 1.0. Иначе - 3.0 или 2.0.
                                base_score = POSITION_SCORES.get(rank, 1.0) 
                                
                                # 2. Определяем тональный множитель
                                multiplier = SENTIMENT_MULTIPLIERS.get(sentiment, 1.0)
                                
                                # 3. Расчет итогового счета
                                final_score = base_score * multiplier

                                # 4. Проверка и сохранение
                                if brand_name_ranked in final_competitors and final_score > 0:
                                    
                                    brand_scores[brand_name_ranked] += final_score
                                    current_query_score += final_score
                                    
                                    detected_brands_details.append({
                                        'brandName': brand_name_ranked,
                                        'sentiment': sentiment,
                                        'score': round(final_score, 2),
                                        'rank': rank # Сохраняем ранг для отображения
                                    })
                                    
                            # Обновляем общий счет, если был засчитан хотя бы один бренд
                            if current_query_score > 0:
                                total_tracked_score += current_query_score

                    except json.JSONDecodeError:
                        st.error(f"Ошибка декодирования JSON при анализе для запроса: {query}")
                    
                
                # Форматируем детали для отчета
                details_text = "\n".join([
                    f"  - {d['brandName']}: Позиция {d['rank']+1}, Тональность '{d['sentiment']}', Счет: {d['score']}"
                    for d in detected_brands_details
                ])
                if not details_text:
                    details_text = "Не найдено или Счет 0"
                    
                # Добавляем детали в отчет
                st.session_state.analysis_details.append({
                    'Запрос': query,
                    'Ответ Gemini': answer_text, 
                    'Анализ (Позиция, Тональность, Счет)': details_text,
                    'Общий Счет Запроса': round(current_query_score, 2)
                })
                
            progress_bar.progress(1.0, text="Анализ завершен!")
            st.success("Анализ Share of Voice завершен!")

            # Формирование финальной таблицы результатов
            final_data = []
            for brand_name_original in final_competitors:
                score = brand_scores.get(brand_name_original, 0.0)
                
                # Расчет SoV
                sov = 0.0
                if total_tracked_score > 0:
                    sov = (score / total_tracked_score) * 100
                
                final_data.append({
                    "Бренд": brand_name_original.strip(),
                    "Итоговый Счет (Total Weighted Score)": round(score, 2),
                    "AI Share of Voice (%)": round(sov, 2)
                })
            
            st.session_state.results = pd.DataFrame(final_data).sort_values(
                by=["Итоговый Счет (Total Weighted Score)", "AI Share of Voice (%)"], 
                ascending=False
            ).reset_index(drop=True)
            st.session_state.step = 5 # Переход к финальному шагу
            st.rerun()


if st.session_state.step == 5 and st.session_state.results is not None:
    
    st.divider()
    
    # --- Шаг 5: Вывод Результатов ---
    st.header("Шаг 5: Результаты AI Share of Voice")
    
    # Выделение вашего бренда
    your_brand_name = st.session_state.brand.strip()
    your_brand_row = st.session_state.results[st.session_state.results["Бренд"] == your_brand_name]
    
    if not your_brand_row.empty:
        st.metric(
            label=f"Ваш AI SoV ({your_brand_name})", 
            value=f'{your_brand_row["AI Share of Voice (%)"].iloc[0]}%'
        )
        
    st.subheader("Сводная Таблица AI SoV")
    st.dataframe(st.session_state.results, use_container_width=True)

    # Пункт 1: Отображение детального отчета
    st.subheader("Подробный Отчет и Ответы Gemini (по данным Шага 4)")
    
    # Обновленная таблица весов для пояснения
    st.markdown("""
    **Система Взвешивания (3-е место и ниже получают 1.0 балл):**
    | Критерий | Позиция (Базовый Счет) | Тональность (Множитель) |
    | :--- | :--- | :--- |
    | **🥇 1-я рекомендация** | 3.0 | Положительная **$\times 1.5$** |
    | **🥈 2-я позиция** | 2.0 | Нейтральная **$\times 1.0$** |
    | **🥉 3-я позиция и ниже** | 1.0 | Отрицательная **$\times 0.0$** |
    """)
    st.caption("Итоговый Счет = Базовый Счет $\times$ Множитель. LLM анализирует до 3 наиболее заметных рекомендаций, применяя базовый счет 1.0 к 3-й позиции (и далее).")
    
    for detail in st.session_state.analysis_details:
        with st.expander(f"Запрос: {detail['Запрос'][:60]}... (Счет: {detail['Общий Счет Запроса']})"):
            st.markdown(f"**Запрос:** `{detail['Запрос']}`")
            st.markdown(f"**Общий Счет Запроса:** `{detail['Общий Счет Запроса']}`")
            st.markdown(f"**Детали Анализа:**")
            st.code(detail['Анализ (Позиция, Тональность, Счет)'], language='markdown')
            st.markdown("---")
            st.markdown("**Полный Ответ Gemini:**")
            st.code(detail['Ответ Gemini'], language='markdown')


# --- Общее Состояние Приложения (Пояснения) ---

if st.session_state.step == 1:
    st.info("Введите название бренда и описание индустрии, чтобы начать.")
elif st.session_state.step == 2:
    st.info("Нажмите кнопку 'Сгенерировать Рекомендательные Запросы' для продолжения.")
elif st.session_state.step == 3:
    st.info("Введите список конкурентов и нажмите 'Получить Ответы Gemini'.")
elif st.session_state.step == 4:
    st.info("Ответы получены. Проверьте сырые ответы ниже и нажмите 'Провести Структурированный Анализ и Расчет SoV'.")
elif st.session_state.step == 5:
    st.success("Анализ завершен! Вы можете просмотреть детали в разделе 'Подробный Отчет'.")

# Футер
st.sidebar.markdown("---")
# Используем условие, чтобы не показывать пустой бренд при первом запуске
if st.session_state.brand:
    st.sidebar.markdown(f"**Ваш Бренд:** `{st.session_state.brand}`")
else:
    st.sidebar.markdown(f"**Ваш Бренд:** *Не задан*")
    
st.sidebar.markdown(f"**Текущий Модель:** `{MODEL_NAME}`")
st.sidebar.markdown(f"**Текущий Шаг:** Шаг {st.session_state.step}")
