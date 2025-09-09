''' Импортируем библиотеки '''
import pandas as pd
import streamlit as st
import pickle


# Список признаков — ДОЛЖЕН СОВПАДАТЬ с теми, на которых обучалась модель!
features = ['Gender_Male', 'Married_Yes', 'TotalApplicantIncome', 'LoanAmount', 'Credit_History']

# ЗАГРУЗКА МОДЕЛИ из .pkl файла
with open('Loan_Approval_Prediction.pkl', 'rb') as f:
    model = pickle.load(f)

# Декоратор для кэширования функции (ускоряет повторные вызовы)
@st.cache_data
def prediction(Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History):   
    
    if Gender == "Мужчина":
        Gender = 0
    else:
        Gender = 1
 
    if Married == "Не в браке":
        Married = 0
    else:
        Married = 1
 
    if Credit_History == "Нет кредитной истории":
        Credit_History = 0
    else:
        Credit_History = 1  
 
    # Приводим LoanAmount к тому же масштабу, что и при обучении (в тысячах)
    LoanAmount = LoanAmount / 1000
 
    # Создаём DataFrame с правильными именами столбцов
    input_data = pd.DataFrame([[Gender, Married, TotalApplicantIncome, LoanAmount, Credit_History]], 
                              columns=features)
    
    # Делаем прогноз
    pred_inputs = model.predict(input_data)
        
    if pred_inputs[0] == 0:
        pred = 'Кредит не одобрен'
    elif pred_inputs[0] == 1:
        pred = 'Кредит одобрен'
    else:
        pred = 'Ошибка'
    return pred


# Главная функция интерфейса
def main():       
    # Заголовок приложения
    html_temp = """ 
    <div style ="background-color:teal;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Приложение Streamlit для прогнозирования одобрения кредита с использованием машинного обучения</h1> 
    </div> 
    """
      
    # Отобразить заголовок
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # Поля ввода для пользователя
    Gender = st.selectbox('Пол', ("Мужчина", "Женщина"))
    Married = st.selectbox('Статус брака', ("Не в браке", "В браке")) 
    ApplicantIncome = st.number_input("Месячный доход (включая созаёмщика, если есть)", min_value=0) 
    LoanAmount = st.number_input("Объём кредита (например: 125000)", min_value=0)
    Credit_History = st.selectbox('Кредитная история', ("Есть кредитная история", "Нет кредитной истории"))
    result = ""
      
    # При нажатии кнопки — делаем прогноз
    if st.button("Спрогнозировать"): 
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) 
        st.success('Финальное решение: {}'.format(result))
        print(LoanAmount)

# Запуск приложения
if __name__ == '__main__': 
    main()