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

# JSON-схема для структурированного позиционного и тонального анализа (Шаг 5)
# Схема остается, но инструкция LLM будет требовать вернуть ВСЕ упомянутые бренды.
SOV_ANALYSIS_SCHEMA = {
    "type": "ARRAY",
    "description": "A ranked list of ALL brands from the provided competitor list that are clearly recommended or mentioned, ordered by their prominence/rank (most prominent/first mention is index 0). Each entry must include the brand name and the associated sentiment.",
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
                # Используем сложную схему, если она предоставлена
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
if 'tracked_brands' not in st.session_state:
    st.session_state.tracked_brands = "" # Изменено имя переменной для ясности
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

st.title("🗣️ AI Share of Voice (SoV) Анализатор (v4.1 - Полный Счет)")
st.markdown("Измерьте, как часто Gemini рекомендует ваш бренд по сравнению с конкурентами.")

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
                            st.session_state.step = 3 # Переход к Шагу 3 (Сбор Ответов)
                            st.success("Запросы сгенерированы! Перейдите к Шагу 3.")
                        else:
                            st.error("Ошибка парсинга: Gemini не вернул корректный JSON-список строк.")
                    except json.JSONDecodeError:
                        st.error(f"Ошибка: Не удалось декодировать JSON. Ответ Gemini: {json_response[:200]}...")
                else:
                    st.error("Не удалось сгенерировать запросы.")

    if st.session_state.step >= 3:
        st.subheader("Финальные Запросы (Отредактируйте при необходимости):")
        # Ensure queries are updated based on user input
        user_queries_input = st.text_area(
            "Список запросов (один запрос на строку):", 
            value=st.session_state.user_queries,
            height=150
        )
        st.session_state.user_queries = user_queries_input # Обновляем состояние после редактирования
        final_queries = [q.strip() for q in st.session_state.user_queries.split('\n') if q.strip()]
        
        st.caption(f"Будет использовано запросов: {len(final_queries)}")
    
    if st.session_state.step >= 3:
        st.divider()

        # --- Шаг 3: Сбор Ответов Gemini ---
        
        st.header("Шаг 3: Сбор Ответов Gemini")
        st.info("Получите ответы Gemini на все запросы, чтобы определить, какие бренды рекомендуются.")
        
        if st.button("Получить Ответы Gemini", disabled=st.session_state.step != 3):
            if not final_queries:
                st.error("Убедитесь, что запросы заполнены в Шаге 2.")
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
                st.success(f"Собрано {len(st.session_state.raw_responses)} ответов. Перейдите к Шагу 4 для определения брендов.")
                st.session_state.step = 4 # Переход к Шагу 4 (Определение Брендов)
                st.rerun()

if st.session_state.step >= 4:
    st.divider()
    
    # --- Шаг 4: Определение и Финальный Список Брендов ---
    st.header("Шаг 4: Определение и Финальный Список Отслеживаемых Брендов")
    st.info("На основе собранных ответов Gemini автоматически извлеките все упомянутые бренды. **Ваш бренд включен по умолчанию.** Отредактируйте список перед анализом.")

    if st.button("Предложить Бренды для Отслеживания (LLM-Извлечение)", disabled=st.session_state.step != 4 or not st.session_state.raw_responses):
        if st.session_state.client and st.session_state.raw_responses:
            with st.spinner("LLM анализирует ответы и извлекает бренды..."):
                
                # Объединяем все ответы в один большой текст для анализа
                full_response_text = " ".join([item['answer'] for item in st.session_state.raw_responses])
                
                # --- ИЗМЕНЕННАЯ СИСТЕМНАЯ ИНСТРУКЦИЯ ДЛЯ ПОВЫШЕНИЯ ТОЧНОСТИ ---
                system_instruction_extraction = (
                    "Вы — аналитик, специализирующийся на извлечении названий брендов из текстов. "
                    "Проанализируйте предоставленные ответы LLM и индустрию. "
                    "Извлеките **только те** уникальные названия брендов, которые упоминаются в ответах как **прямые или текущие рекомендации, или современные альтернативы** в индустрии "
                    f"'{st.session_state.industry}'. **Игнорируйте** бренды, упомянутые исключительно в историческом, контекстном, или сравнительном разрезе (например, 'ушедшие с рынка'). "
                    "Выведите только JSON-список строк (названий брендов)."
                )
                # -----------------------------------------------------------------
                
                extraction_prompt = f"Ответы LLM: '''{full_response_text}'''"
                
                json_extraction_response = generate_content_with_retry(
                    st.session_state.client, 
                    extraction_prompt, 
                    system_instruction=system_instruction_extraction, 
                    json_output=True
                )
                
                # Инициализируем извлеченные бренды с вашим брендом
                unique_brands_set = {st.session_state.brand.strip()}
                
                if json_extraction_response:
                    try:
                        extracted_brands = json.loads(json_extraction_response)
                        if isinstance(extracted_brands, list) and all(isinstance(b, str) for b in extracted_brands):
                            # Добавляем все извлеченные бренды
                            for b in extracted_brands:
                                if b.strip():
                                    unique_brands_set.add(b.strip())
                            
                            st.session_state.tracked_brands = ", ".join(sorted(list(unique_brands_set)))
                            st.success("Бренды извлечены. Отредактируйте список ниже.")
                        else:
                            st.error("Ошибка парсинга: LLM не вернул корректный JSON-список строк. Попробуйте ввести вручную.")
                            st.session_state.tracked_brands = st.session_state.brand
                    except json.JSONDecodeError:
                        st.error(f"Ошибка: Не удалось декодировать JSON. Ответ LLM: {json_extraction_response[:200]}... Попробуйте ввести вручную.")
                        st.session_state.tracked_brands = st.session_state.brand
                else:
                    st.error("Не удалось извлечь бренды. Попробуйте ввести вручную.")
                    st.session_state.tracked_brands = st.session_state.brand


    st.subheader("Финальный Список Отслеживаемых Брендов (Отредактируйте):")
    
    # Текстовое поле для редактирования списка брендов
    st.session_state.tracked_brands = st.text_area(
        "Список Брендов (через запятую):", 
        value=st.session_state.tracked_brands if st.session_state.tracked_brands else st.session_state.brand,
        height=100,
        help="Отредактируйте список, чтобы оставить только те бренды, которые вы хотите включить в анализ SoV. Ваш бренд (YOUR_BRAND_NAME) должен быть включен."
    )
    
    final_competitors = [c.strip() for c in st.session_state.tracked_brands.split(',') if c.strip()]
    
    # Финальная проверка: ваш бренд должен быть в списке
    if st.session_state.brand.strip() not in final_competitors:
        st.warning(f"Ваш бренд '{st.session_state.brand}' не найден в списке. Он будет добавлен.")
        final_competitors.append(st.session_state.brand.strip())
        final_competitors = list(set(final_competitors))

    st.caption(f"Будет отслеживаться брендов: {len(final_competitors)}")

    if st.button("Подтвердить Список и Перейти к Анализу SoV", disabled=st.session_state.step != 4 or not final_competitors):
        if len(final_competitors) > 0:
            st.session_state.step = 5 # Переход к Шагу 5 (Анализ)
            st.rerun()
        else:
            st.error("Список брендов не может быть пустым.")


if st.session_state.step >= 5:
    # Убедимся, что final_competitors определен, если мы перешли сюда
    final_competitors = [c.strip() for c in st.session_state.tracked_brands.split(',') if c.strip()]
    if st.session_state.brand.strip() not in final_competitors:
        final_competitors.append(st.session_state.brand.strip())
        final_competitors = list(set(final_competitors))
        
    st.divider()
    
    # --- Шаг 5: Структурированный Анализ и Расчет AI SoV ---

    st.header("Шаг 5: Структурированный Анализ AI SoV (Полный Счет)")
    st.info(f"Будет проанализировано {len(st.session_state.raw_responses)} ответов с использованием {len(final_competitors)} брендов из финального списка.")
    
    # --- Отображение сырых ответов (Данные для Шага 5) ---
    if st.session_state.raw_responses:
        st.subheader("Данные для Анализа (Сырые Ответы из Шага 3)")
        st.caption("Проверьте эти ответы. Анализ LLM будет проведен на основе этого текста.")
        for i, item in enumerate(st.session_state.raw_responses):
            with st.expander(f"Ответ {i+1}: {item['query'][:60]}..."):
                st.code(item['answer'], language='markdown')


    if st.button("Провести Структурированный Анализ и Расчет SoV", disabled=st.session_state.step != 5 or not st.session_state.raw_responses):
        if not final_competitors:
            st.error("Убедитесь, что конкуренты заполнены в Шаге 4.")
        elif st.session_state.client and st.session_state.raw_responses:
            
            # --- КОНСТАНТЫ СЧЕТА И МНОЖИТЕЛЕЙ ---
            # Базовый позиционный балл (Position Score)
            # 3-е место и все последующие получают базовый балл 1.0 (Fixed: use .get(rank, 1.0))
            POSITION_SCORES = {
                0: 3.0, # 1st place
                1: 2.0, # 2nd place
                2: 1.0, # 3rd place 
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
                    "Ваша задача — определить, **все** бренды из 'СПИСОК_БРЕНДОВ', которые упоминаются в тексте. "
                    "Верните полный список упомянутых брендов, **ранжированный по их заметности или порядку упоминания** (самый заметный/первый в списке должен быть на позиции 1). "
                    "Для каждого бренда определите тональность упоминания (Positive, Neutral, или Negative). "
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
                                # Если ранг >= 2 (3-е место или ниже), используем 1.0. 
                                # Иначе - 3.0 (rank 0) или 2.0 (rank 1).
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
            st.session_state.step = 6 # Переход к финальному шагу
            st.rerun()


if st.session_state.step == 6 and st.session_state.results is not None:
    
    st.divider()
    
    # --- Шаг 6: Вывод Результатов ---
    st.header("Шаг 6: Результаты AI Share of Voice")
    
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
    st.subheader("Подробный Отчет и Ответы Gemini (по данным Шага 5)")
    
    # Обновленная таблица весов для пояснения
    st.markdown("""
    **Система Взвешивания (3-е место и ниже получают 1.0 балл):**
    | Критерий | Позиция (Базовый Счет) | Тональность (Множитель) |
    | :--- | :--- | :--- |
    | **🥇 1-я рекомендация** | 3.0 | Положительная **$\times 1.5$** |
    | **🥈 2-я позиция** | 2.0 | Нейтральная **$\times 1.0$** |
    | **🥉 3-я позиция и ниже** | 1.0 | Отрицательная **$\times 0.0$** |
    """)
    st.caption("Итоговый Счет = Базовый Счет $\times$ Множитель. LLM анализирует **все** упомянутые бренды, присваивая базовый счет 1.0 всем позициям, начиная с 3-й.")
    
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
    st.info("Нажмите 'Получить Ответы Gemini'. Это соберет данные для анализа.")
elif st.session_state.step == 4:
    st.info("Нажмите 'Предложить Бренды для Отслеживания', чтобы LLM извлек все упомянутые бренды из ответов. Затем отредактируйте список, который будет использоваться для анализа.")
elif st.session_state.step == 5:
    st.info("Список брендов готов. Нажмите 'Провести Структурированный Анализ и Расчет SoV'.")
elif st.session_state.step == 6:
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
